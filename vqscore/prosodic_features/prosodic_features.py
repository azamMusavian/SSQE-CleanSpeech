import os
import math
import torch
import torchaudio
import torchaudio.functional as F

from torch.nn.functional import conv1d, pad

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from .utils import lookup_or_compute_pitch, compute_log_energy, cepstral_flux, \
    kill_bleeding, percentilize_pitch, find_cluster_means, smooth_jcc, \
    epeakness, ppeakness, misalignment, rectangular_filter, smooth, \
    z_normalize, windowize
import scipy.stats as st


def compute_windowed_slips(energy, pitch, duration):
    if len(energy) == len(pitch) + 1:
        energy = energy[:-1]
    elif len(energy) != len(pitch):
        print(f'length(energy) is {len(energy)} but length(pitch) is {len(pitch)}')
        return None

    epeaky = epeakness(energy)
    ppeaky = ppeakness(pitch)
    misa = misalignment(epeaky, ppeaky)
    smoothed = smooth(misa, rectangular_filter(duration))
    return smoothed


def compute_rate(log_energy: torch.Tensor, window_size_ms: int) -> torch.Tensor:
    """
    Computes a speaking-rate proxy using a proxy for spectral flux.

    Parameters:
    log_energy (torch.Tensor): Log energy values per frame
    window_size_ms (int): Window size in milliseconds

    Returns:
    torch.Tensor: Scaled liveliness values per frame
    """
    frames = window_size_ms // 10

    # Compute inter-frame deltas
    deltas = torch.abs(log_energy[1:] - log_energy[:-1])
    cum_sum_deltas = torch.cat((torch.tensor([0]), torch.cumsum(deltas, dim=0)))

    # Compute windowed liveliness
    window_liveliness = cum_sum_deltas[(frames - 1):] - cum_sum_deltas[:-(frames - 1)]

    # Normalize rate for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(log_energy)  # k-means algorithm
    scaled_liveliness = (window_liveliness - silence_mean) / (speech_mean - silence_mean)

    # Padding head and tail frames
    head_frames_to_pad = (frames // 2) - 1
    tail_frames_to_pad = (frames + 1) // 2 - 1
    scaled_liveliness = torch.cat((torch.zeros(head_frames_to_pad), scaled_liveliness, torch.zeros(tail_frames_to_pad)))
    return scaled_liveliness


def compute_lengthening(relevant_energy, relevant_flux, duration):
    """Compute lengthening based on relevant energy and flux."""
    relevant_flux = relevant_flux.float()

    non_nan_flux = relevant_flux[~torch.isnan(relevant_flux)]
    non_nan_mean = non_nan_flux.mean()
    non_nan_std = non_nan_flux.std()

    relevant_flux[torch.isnan(relevant_flux)] = non_nan_mean

    max_plausible = non_nan_mean + 3 * non_nan_std
    relevant_flux[relevant_flux > max_plausible] = max_plausible

    if 0 in relevant_flux:
        relevant_flux[relevant_flux == 0] = non_nan_mean / (3 * non_nan_std)

    # Azam: Added the 3 below lines to fix the tensor mismatch issue
    min_length = min(relevant_energy.shape[0], relevant_flux.shape[0])
    relevant_energy = relevant_energy[:min_length]
    relevant_flux = relevant_flux[:min_length]

    lengthening = relevant_energy / relevant_flux
    lengthening[torch.isinf(lengthening)] = 0
    lengthening[torch.isnan(lengthening)] = 0

    return smooth_jcc(lengthening, duration)


def compute_CPPS(signal, sample_rate):
    """Compute Smoothed Cepstral Peak Prominence (CPPS)."""

    # Window size and step
    win_step_s = 0.002  # Step size in seconds
    win_len = round(0.04 * sample_rate)
    win_step = round(win_step_s * sample_rate)
    win_overlap = win_len - win_step

    # Quefrency range
    quef_bot = round(sample_rate / 300)
    quef_top = round(sample_rate / 60)
    quefs = torch.arange(quef_bot, quef_top + 1) - 1

    # Pre-emphasis from 50 Hz
    signal = F.preemphasis(signal, 0.5)

    # Compute spectrum and cepstrum
    # spec will have length equal to
    # fix((s_len - win_overlap)/(win_len - win_overlap))
    spec = F.spectrogram(signal,
                         pad=0,
                         window=torch.hann_window(win_len),
                         n_fft=512,
                         win_length=win_len,
                         hop_length=win_step,
                         power=2,
                         normalized=True,
                         )
    spec_log = 10 * torch.log10(spec + 1e-10)
    ceps_log = 10 * torch.log10(torch.fft.fft(spec_log, dim=0).real ** 2 + 1e-10)

    # Do time- and quefrency-smoothing
    # Smooth over 10 samples and 10 quefrency bins
    smooth_filt_b = torch.ones(1, 1, 10) / 10

    ch = spec.size(0)
    nframes = spec.size(1)
    ceps_log = conv1d(
        conv1d(ceps_log, smooth_filt_b.expand(ch, -1, -1), padding=9, groups=ch)[:, :-9].t(),
        smooth_filt_b.expand(nframes, -1, -1), padding=9, groups=nframes
    )[:, :-9].t().contiguous()

    # Find cepstral peaks in the quefrency range
    ceps_log = ceps_log[quefs, :]
    peak, peak_quef = ceps_log.max(0)

    # Get the regression line and calculate its distance from the peak
    n_wins = ceps_log.shape[1]
    ceps_norm = torch.zeros(n_wins)
    for n in range(n_wins):
        p = np.polyfit(quefs, ceps_log[:, n], 1)
        ceps_norm[n] = np.polyval(p, quefs[peak_quef[n]])

    cpps = peak - ceps_norm

    # Pad the CPPS vector and calculate means in 10-ms window
    midlevelFrameWidth_ms = 10
    midlevelFrameWidth_s = midlevelFrameWidth_ms / 1000

    # Reshape cpps so that each column represents 10 ms. Since win_step_s
    # and midlevelFrameWidth_s are fixed, cppsReshapeNumCols is always 5,
    # but is added here for clarity.
    cppsReshapeNumCols = round(midlevelFrameWidth_s / win_step_s)

    signalDuration_s = len(signal) / sample_rate
    signalDuration_ms = signalDuration_s * 1000
    expectedCPPSmidlevelLen = int(signalDuration_ms / midlevelFrameWidth_ms)
    totalPaddingSize = expectedCPPSmidlevelLen * cppsReshapeNumCols - cpps.shape[0]
    if totalPaddingSize > 0:
        prepadSize = totalPaddingSize // 2
        postpadSize = totalPaddingSize - prepadSize

        cpps_padded = pad(cpps, (prepadSize, postpadSize), value=torch.nan)
    else:
        totalPaddingSize = abs(totalPaddingSize)
        pre = totalPaddingSize // 2
        post = totalPaddingSize - pre
        cpps_padded = cpps[pre:-post]

    cpps_win = cpps_padded.reshape(-1, cppsReshapeNumCols)
    CPPS_midlevel, _ = torch.nanmedian(cpps_win, 1)

    # replace NaNs with median CPPS
    CPPS_midlevel[torch.isnan(CPPS_midlevel)] = torch.nanmedian(CPPS_midlevel)
    return CPPS_midlevel


def compute_pitch_range(pitch, window_size, range_type):
    """Compute evidence for different types of pitch ranges."""
    ms_per_window = 10
    frames_per_window = window_size / ms_per_window
    relevant_span = 1000
    frames_per_half_span = int((relevant_span / 2) / frames_per_window)

    range_count = torch.zeros(len(pitch))
    for i in range(len(pitch)):
        # get offset of 500 ms
        start_neighbors = max(i - frames_per_half_span, 0)
        end_neighbors = min(i + frames_per_half_span, len(pitch))
        neighbors = pitch[start_neighbors:end_neighbors]
        ratios = neighbors / pitch[i]

        if range_type == 'f':
            range_count[i] = torch.sum((ratios >= 0.99) & (ratios <= 1.01))
        elif range_type == 'n':
            range_count[i] = torch.sum((ratios > 0.98) & (ratios < 1.02))
        elif range_type == 'w':
            range_count[i] = torch.sum(((ratios > 0.70) & (ratios < 0.90)) | ((ratios > 1.1) & (ratios < 1.3)))

    integral_image = torch.cat((torch.tensor([0]), torch.cumsum(range_count, 0)))
    window_values = integral_image[int(frames_per_window):] - integral_image[:-int(frames_per_window)]

    padding_needed = frames_per_window - 1
    front_padding = torch.zeros((math.floor(padding_needed / 2),))
    tail_padding = torch.zeros((math.ceil(padding_needed / 2),))
    return torch.cat((front_padding, window_values, tail_padding)) / frames_per_window


def compute_pitch_in_band(percentiles, band_flag, window_size_ms):
    """Compute evidence for pitch being in a specified band."""
    frames_per_window = window_size_ms // 10
    percentiles = torch.nan_to_num(percentiles, nan=0.00 if band_flag in ['h', 'th'] else 1.00)

    if band_flag == 'h':
        evidence_vector = percentiles
    elif band_flag == 'l':
        evidence_vector = 1 - percentiles
    elif band_flag == 'th':
        percentiles[percentiles < 0.50] = 0.50
        evidence_vector = percentiles - 0.50
    elif band_flag == 'tl':
        percentiles[percentiles > 0.50] = 0.50
        evidence_vector = 0.50 - percentiles
    else:
        raise ValueError(f"Unknown band flag: {band_flag}")

    integral_image = torch.cat((torch.tensor([0]), torch.cumsum(evidence_vector, 0)))
    window_values = integral_image[frames_per_window:] - integral_image[:-frames_per_window]
    padding_needed = frames_per_window - 1
    front_padding = torch.zeros((math.floor(padding_needed / 2),))
    tail_padding = torch.zeros((math.ceil(padding_needed / 2),))
    return torch.cat((front_padding, window_values, tail_padding)) / frames_per_window


def compute_creakiness(pitch, window_size_ms):
    """Compute evidence of creakiness in pitch variations."""
    frames_per_window = window_size_ms // 10
    ratios = pitch[1:] / pitch[:-1]

    octave_up = (ratios > 1.90) & (ratios < 2.10)
    octave_down = (ratios > 0.475) & (ratios < 0.525)
    small_up = (ratios > 1.05) & (ratios < 1.25)
    small_down = (ratios < 0.95) & (ratios > 0.80)

    creakiness = octave_up + octave_down + small_up + small_down
    integral_image = torch.cat((torch.tensor([0]), torch.cumsum(creakiness, 0)))
    creakiness_per_window = integral_image[frames_per_window:] - integral_image[:-frames_per_window]

    head_padding = torch.zeros((frames_per_window // 2,))
    tail_padding = torch.zeros((frames_per_window // 2,))

    creak_values = torch.cat((head_padding, creakiness_per_window, tail_padding)) / frames_per_window
    return creak_values


def window_energy(log_energy, ms_per_window):
    """Compute windowed energy from log energy."""
    integral_image = torch.cat([torch.tensor([0]), torch.cumsum(log_energy, 0)])
    frames_per_window = ms_per_window // 10
    window_sum = integral_image[frames_per_window:] - integral_image[:-frames_per_window]
    silence_mean, speech_mean = find_cluster_means(window_sum)
    difference = speech_mean - silence_mean

    if difference > 0:
        scaled_sum = (window_sum - silence_mean) / difference
    else:
        scaled_sum = (window_sum - (0.5 * silence_mean)) / silence_mean

    head_pad = torch.zeros(int(np.floor(frames_per_window / 2) - 1))
    tail_pad = torch.zeros(int(np.ceil(frames_per_window / 2) - 1))
    return torch.concatenate((head_pad, scaled_sum, tail_pad))


def make_track_monster(fn, featurelist, msPerFrame=10, use_5sec=False):
    signalPair, rate = torchaudio.load(fn)
    # Resample at 16000Hz because higher rates seem to confuse the pitch tracker
    signalPair = F.resample(waveform=signalPair, orig_freq=rate, new_freq=8000)
    rate = 8000

    # Add 5-second selection only if use_5sec is True
    if use_5sec:
        samples_5sec = 5 * rate  # 5 seconds at 8000 Hz = 40000 samples
        data_shape = signalPair.shape[1]

        if data_shape <= samples_5sec:
            # If audio is shorter than 5 seconds, pad with zeros
            padding = torch.zeros(signalPair.shape[0], samples_5sec - data_shape)
            signalPair = torch.cat((signalPair, padding), dim=1)
        else:
            # Use deterministic selection instead of random
            start = deterministic_start_index(fn, data_shape, samples_5sec)
            signalPair = signalPair[:, start:start + samples_5sec]

    processAudio = False
    firstCompleteFrame = 1
    lastCompleteFrame = float('inf')

    for feature in featurelist:
        featname = feature['featname']
        if featname in ['vo', 'th', 'tl', 'lp', 'hp', 'fp', 'wp', 'np', 'sr', 'cr', 'pd', 'le', 'vf', 'sf', 're', 'en',
                        'ts', 'te']:
            processAudio = True

    if processAudio:
        stereop = True if signalPair.shape[0] == 2 else False
        if signalPair.shape[0] < 2 and stereop:
            # raise ValueError(f"{trackspec['path']} is not a stereo file but expected to be.")
            raise ValueError(f"{fn} is not a stereo file but expected to be.")

        samplesPerFrame = msPerFrame * (rate / 1000)
        signall = signalPair[0]
        plraw, pCenters = lookup_or_compute_pitch(signall, rate)
        energyl = compute_log_energy(signall, samplesPerFrame)

        # --- FIX (Azam): Ensure energy and pitch have the same length ---
        min_len = min(len(plraw), len(energyl))
        plraw = plraw[:min_len]
        energyl = energyl[:min_len]
        # ---------------------------------------------------------

        pitchl = plraw
        cepstralFluxl = cepstral_flux(signall, rate, energyl)
        cppsl = z_normalize(compute_CPPS(signall, rate))  # should use lookupOrComputeCpps

        if stereop:
            signalr = signalPair[1]
            prraw, _ = lookup_or_compute_pitch(signalr, rate)
            energyr = compute_log_energy(signalr, samplesPerFrame)
            cepstralFluxr = cepstral_flux(signalr, rate, energyr)
            cppsr = z_normalize(compute_CPPS(signalr, rate))  # should use lookupOrComputeCpps

            pitchl, pitchr, npoints = kill_bleeding(plraw, prraw, energyl, energyr)
            pitchl, pitchr, energyl, energyr = pitchl[:npoints], pitchr[:npoints], energyl[:npoints], energyr[:npoints]
            cepstralFluxl, cepstralFluxr = cepstralFluxl[:npoints], cepstralFluxr[:npoints]

        nframes = int(len(signalPair[0]) // samplesPerFrame)
        lastCompleteFrame = min([nframes, lastCompleteFrame, npoints] if stereop else [nframes, lastCompleteFrame])

    maxPitch = 500
    pitchLper = percentilize_pitch(pitchl, maxPitch)
    if stereop:
        pitchRper = percentilize_pitch(pitchr, maxPitch)

    features_array = []

    for feature in featurelist:
        feattype = feature['featname']
        duration = 600  # signalPair.shape[1]
        side = 'self'

        if processAudio:
            relevantEnergy, relevantPitch, relevantPitchPer, relevantFlux, relevantCpps = (
                energyl, pitchl, pitchLper, cepstralFluxl, cppsl) if side == 'self' else (
                energyr, pitchr, pitchRper, cepstralFluxr, cppsr)

        featurevec = np.zeros(lastCompleteFrame - 1)
        if feattype == 'sr':  # speaking rate
            featurevec = compute_rate(relevantEnergy, duration)
        #            print(feattype, relevantEnergy.shape, featurevec.shape)
        elif feattype == 'le':  # lengthening
            featurevec = compute_lengthening(relevantEnergy, relevantFlux, duration)
        #            print(feattype, featurevec.shape)
        elif feattype == 'pd':  # peakDisalignment
            featurevec = compute_windowed_slips(relevantEnergy, relevantPitchPer, duration)
        #            print(feattype, featurevec.shape)
        elif feattype == 'cp':  # CPPS
            featurevec = windowize(relevantCpps, duration)
        #            print(feattype, featurevec.shape)
        elif feattype == 'lp':  # pitch lowness     CHECK
            featurevec = compute_pitch_in_band(relevantPitchPer, 'l', duration)
        #            print(feattype, featurevec.shape)
        elif feattype == 'hp':  # pitch highness      CHECK
            featurevec = compute_pitch_in_band(relevantPitchPer, 'h', duration)
        #            print(feattype, featurevec.shape)
        elif feattype == 'np':  # narrow pitch range  CHECK
            featurevec = compute_pitch_range(relevantPitch, duration * 2, 'n')
        #            print(feattype, featurevec.shape)
        elif feattype == 'wp':  # wide pitch range    CHECK
            featurevec = compute_pitch_range(relevantPitch, duration, 'w')
        #            print(feattype, featurevec.shape)
        elif feattype == 'cr':  # creakiness
            featurevec = compute_creakiness(relevantPitch, duration);
        #            print(feattype, featurevec.shape)
        elif feattype == 'vo':  # intensity
            featurevec = window_energy(relevantEnergy, duration)
        #            print(feattype, featurevec.shape)
        shift = round((feature['startms'] + feature['endms']) / (2 * msPerFrame))

        if shift < 0:
            shifted = torch.cat((torch.zeros(-shift), featurevec[:shift]))
        else:
            shifted = torch.cat((featurevec[shift:], torch.zeros(shift)))

        shifted = shifted[:lastCompleteFrame - 1]
        features_array.append(shifted)

    monster = torch.column_stack(features_array)
    return monster


def deterministic_start_index(filename, audio_length, target_length):
    """
    Generate a deterministic start index based on the filename.
    Both CNN feature extraction and prosodic feature extraction will use this.
    """
    # Use a hash of the filename to generate a reproducible "random" index
    import hashlib

    # Get a hash of the filename
    filename_hash = hashlib.md5(filename.encode()).hexdigest()

    # Convert first 8 characters of hash to integer
    hash_int = int(filename_hash[:8], 16)

    # Scale to get an index within the valid range
    if audio_length <= target_length:
        return 0  # No need for selection if audio is shorter
    else:
        # Get a value in range [0, audio_length - target_length]
        valid_range = audio_length - target_length
        return hash_int % (valid_range + 1)


def extract_prosodic_features(audio_file, startms=-200, endms=200, use_5sec= False):
    duration = endms - startms
    # feature_list = [
    #     {'featname': 'sr', 'startms': startms, 'endms': endms},
    #     {'featname': 'le', 'startms': startms, 'endms': endms},
    #     {'featname': 'pd', 'startms': startms, 'endms': endms},
    #     {'featname': 'cp', 'startms': startms, 'endms': endms},
    #     {'featname': 'lp', 'startms': startms, 'endms': endms},  # wrong
    #     {'featname': 'hp', 'startms': startms, 'endms': endms},  # wrong
    #     {'featname': 'np', 'startms': startms, 'endms': endms},  # wrong
    #     {'featname': 'wp', 'startms': startms, 'endms': endms},  # wrong
    #     {'featname': 'cr', 'startms': startms, 'endms': endms},
    #     {'featname': 'vo', 'startms': startms, 'endms': endms},
    # ]

    feature_list = [
        {'featname': 'sr', 'startms': startms, 'endms': endms},  # speaking rate (43.7%)
        {'featname': 'le', 'startms': startms, 'endms': endms},  # lengthening (20.9%)
        {'featname': 'pd', 'startms': startms, 'endms': endms},  # peak disalignment (7.8%)
        {'featname': 'cp', 'startms': startms, 'endms': endms},  # CPPS (5.9%)
        {'featname': 'hp', 'startms': startms, 'endms': endms},  # pitch highness (4.1%)
        # {'featname': 'np', 'startms': startms, 'endms': endms},  # pitch narrowness (4.0%)
        # {'featname': 'wp', 'startms': startms, 'endms': endms},  # pitch wideness (3.9%)
        # {'featname': 'cr', 'startms': startms, 'endms': endms},  # creakiness (3.5%)
        # {'featname': 'lp', 'startms': startms, 'endms': endms},  # pitch lowness (3.3%)
        # {'featname': 'vo', 'startms': startms, 'endms': endms},  # intensity (3.1%)
    ]

    feat = make_track_monster(audio_file, feature_list, duration, use_5sec=use_5sec)
    feat_stat = torch.zeros(6, 5)
    feat_stat[0] = feat.mean(0)
    feat_stat[1] = feat.std(0)
    feat_stat[2] = feat.max(0)[0]
    feat_stat[3] = feat.min(0)[0]
    feat_stat[4] = torch.from_numpy(st.skew(feat, 0))
    feat_stat[5] = torch.from_numpy(st.kurtosis(feat, 0))

    # Replace NaN values with zeros
    feat_stat[torch.isnan(feat_stat)] = 0
    return feat, feat_stat

