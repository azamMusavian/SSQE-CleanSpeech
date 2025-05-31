import math
import torch
import torchaudio.transforms as T
import torchaudio.functional as F

from torch.nn.functional import conv1d

from .fxrapt import fxrapt


def percentilize_pitch(pitch_points, max_pitch):

    rounded = torch.round(pitch_points)
    
    counts = torch.zeros(max_pitch + 1, dtype=int)
    for pitch in rounded:
        if 0 <= pitch <= max_pitch:
            counts[int(pitch)] += 1
    
    cumulative_sum = torch.cumsum(counts, 0)
    mapping = cumulative_sum / cumulative_sum[max_pitch]
    
    percentiles = torch.full_like(rounded, torch.nan)
    for i, pitch in enumerate(rounded):
        if 0 <= pitch <= max_pitch:
            percentiles[i] = mapping[int(pitch)]
    
    return percentiles


def laplacian_of_gaussian(sigma):
    # if longer, slows things down a little in the subsequent convolution
    # if shorter, insufficient consideration of local context to see if it's a real peak 
    length = sigma * 5

    sigmaSquare = sigma * sigma
    sigmaFourth = sigmaSquare * sigmaSquare

    vec = torch.zeros(length)
    center = math.floor(length / 2)
    for i in range(length):
        x = (i+1) - center;
        y = ((x * x) / sigmaFourth - 1 / sigmaSquare) * math.exp( (-x * x) / (2 * sigmaSquare))
        vec[i] = - y
    return vec


def rectangular_filter(windowDurationMs):
    durationFrames = math.floor(windowDurationMs / 10)
    filter_kernel = torch.ones(durationFrames) / durationFrames
    return filter_kernel


def triangle_filter(windowDurationMs):
    """
    returns a filter to be convolved with a pitch-per-frame vector etc.
    """
    durationFrames = math.floor(windowDurationMs / 10)
    center = math.floor(durationFrames / 2)
    filter_kernel = torch.zeros(durationFrames)
    for i in range(durationFrames):
        filter_kernel[i] = center - abs((i+1) - center)
  
    filter_kernel /= filter_kernel.sum()   # normalize it to sum to one
    return filter_kernel


def z_normalize(vec):
    return (vec - vec.mean()) / vec.std()


def smooth(signal, filter_kernel):
    return conv1d(signal[None], filter_kernel[None].fliplr()[None,:], padding='same')[0]


def myconv(vec, filter_kernel, filterHalfWidth):
    result = conv1d(vec[None], filter_kernel[None].fliplr()[None,:], padding='same')[0]
    trimWidth = math.floor(filterHalfWidth)
    result[:trimWidth] = 0
    result[-trimWidth:] = 0
    return result


def epeakness(vec):
    iSFW = 6  # in-syllable filter width, in frames
    iFFW = 15 # in-foot filter width, in frames
    height = (vec - vec.min()) / (vec.max() - vec.min()).sqrt()
    inSyllablePeakness = myconv(vec, laplacian_of_gaussian(iSFW), iSFW * 2.5)
    inFootPeakness     = myconv(vec, laplacian_of_gaussian(iFFW), iFFW * 2.5)

    peakness = inSyllablePeakness * inFootPeakness * height
    peakness[peakness<0] = 0  # since we don't consider troughs when aligning
    return peakness


def ppeakness(pitchPtile):
    ssFW = 10;    # stressed-syllable filter width; could be 12

    validPitch = pitchPtile > 0
    localPitchAmount = myconv(1.0 * validPitch, triangle_filter(160), 10);
    pitchPtile[pitchPtile.isnan()] = 0
    localPeakness = myconv(pitchPtile, laplacian_of_gaussian(ssFW), 2.5 * ssFW);

    peakness = localPeakness * localPitchAmount * pitchPtile
    peakness[peakness<0] = 0      # don't care about troughs
    return peakness


def misalignment(epeaky, ppeaky):
    """
    % inspired by the need to estimate pitch-peak delay, 
    % but a much simpler conception: just measure misalignment
    % don't worry whether it's pulled forward or back
    % because who can tell? without a stressed-syllable oracle
    % note that a more traditional, but less reliable, estimator is computeSlip.m
    % 
    % Note that misalignments are only salient if they come at a peak.
    % see also comments in ../sliptest/README.TXT
    %
    % note that "misalignment" is a misnomer, since these things are 
    %  not errors; perhaps "disalignment"
    """
    def find_local_max(vector, widthMs):
        """
        % Return a vector where each element e is 
        %  the max value found a window of size width centered about position e.
        % Maybe should have a discount factor for elements further off,
        %  or otherwise soften the edges of this filter
        """
        halfwidthFrames = (widthMs / 2) / 10
        mx = torch.zeros(vector.nelement())
        for e in range(vector.nelement()):
            startframe = max(0, int(e - halfwidthFrames))
            endframe = min(int(e + halfwidthFrames + 1), vector.nelement())
            mx[e] = vector[startframe:endframe].max()
        return mx

    localMaxEPeak = find_local_max(epeaky, 120)

    expectedProduct = localMaxEPeak * ppeaky
    actualProduct = epeaky * ppeaky  

    estimate = (expectedProduct - actualProduct) * ppeaky
    return estimate


def lookup_or_compute_pitch(signal, rate):
    """Return a vector of pitch points and a vector of where they are, in ms."""
    ms_per_sample = 1000 / rate
    pitch, starts_and_ends = fxrapt(signal, rate, 'u')
    pitch_centers = 0.5 * (starts_and_ends[:, 0] + starts_and_ends[:, 1]) * ms_per_sample
    padded_pitch = torch.cat([torch.full((1,), torch.nan), pitch, torch.full((1,), torch.nan)])
    padded_centers = torch.cat([pitch_centers[:1] - 10, pitch_centers, pitch_centers[-1][None] + 10])
    return padded_pitch, padded_centers


def compute_pitch(signal, rate):
    waveform = signal.unsqueeze(0)
    pitch = F.detect_pitch_frequency(waveform, rate).flatten()
    starts_and_ends = torch.column_stack((torch.arange(len(pitch)) * 10, (torch.arange(len(pitch)) + 1) * 10))
    return pitch, starts_and_ends


def kill_bleeding(pitchl, pitchr, energyl, energyr):
    """
    If the pitch is the same in both tracks and one track is significantly louder,
    assume bleeding and set the pitch value in the quieter track to NaN.
    """
    clear_difference = 0.8
    
    # Adjust energy features
    energyl_shifted = energyl[:-1] + energyl[1:]
    energyr_shifted = energyr[:-1] + energyr[1:]
    
    # Determine the minimum number of pitch points
    n_pitch_points = min(len(pitchl), len(pitchr), len(energyl_shifted), len(energyr_shifted))
    
    pitchl = pitchl[:n_pitch_points]
    pitchr = pitchr[:n_pitch_points]
    energyl_shifted = energyl_shifted[:n_pitch_points]
    energyr_shifted = energyr_shifted[:n_pitch_points]
    
    left_louder = energyl_shifted > energyr_shifted + clear_difference
    right_louder = energyr_shifted > energyl_shifted + clear_difference
    
    pitches_same = (pitchl / pitchr > 0.95) & (pitchl / pitchr < 1.05)
    pitches_doubled = (pitchl / pitchr > 0.475) & (pitchl / pitchr < 0.525)
    pitches_halved = (pitchl / pitchr > 1.90) & (pitchl / pitchr < 2.10)
    pitches_suspect = pitches_same | pitches_doubled | pitches_halved
    
    bleeding_to_left = right_louder & pitches_suspect
    bleeding_to_right = left_louder & pitches_suspect

    clean_pitchl = pitchl.clone()
    clean_pitchr = pitchr.clone()
    clean_pitchl[bleeding_to_left] = torch.nan
    clean_pitchr[bleeding_to_right] = torch.nan
    
    return clean_pitchl, clean_pitchr, n_pitch_points


def compute_log_energy(signal: torch.Tensor, samples_per_window: int) -> torch.Tensor:
    """
    Computes the log energy of a signal with non-overlapping frames.
    
    Parameters:
    signal (torch.Tensor): Input signal (1D tensor)
    samples_per_window (int): Number of samples per frame (window size)
    
    Returns:
    torch.Tensor: Log energy values per frame
    """
    # Ensure signal is a floating point tensor

    # Square the signal values
    squared_signal = signal ** 2
    
    # Compute the cumulative sum
    integral_image = torch.cat((torch.tensor([0.0]), torch.cumsum(squared_signal, dim=0)))
    
    # Extract integral values at frame positions
    frame_indices = torch.arange(0, len(integral_image), samples_per_window, dtype=int)
    integral_image_by_frame = integral_image[frame_indices]
    
    # Compute per-frame energy
    per_frame_energy = integral_image_by_frame[1:] - integral_image_by_frame[:-1]
    per_frame_energy = torch.sqrt(per_frame_energy)
    
    # Replace zeros with a small positive value to prevent log(0)
    per_frame_energy = torch.where(per_frame_energy == 0, torch.tensor(1.0), per_frame_energy)
    
    # Compute log energy
    log_energy = torch.log(per_frame_energy)
    
    return log_energy


def average_of_near_values(values, near_mean, far_mean):
    """Returns the average of all points closer to near_mean than to far_mean."""
    # Handle empty tensor case
    if values.numel() == 0:
        return near_mean

    nsamples = 2000
    if values.numel() < nsamples:
        samples = values
    else:
        indices = torch.linspace(0, values.nelement() - 1, steps=nsamples).long()
        samples = values[indices]

    closer_samples = samples[torch.abs(samples - near_mean) < torch.abs(samples - far_mean)]

    if closer_samples.numel() == 0:
        return 0.9 * near_mean + 0.1 * far_mean
    else:
        return closer_samples.mean()


def find_cluster_means(values):
    """Finds the centers of two clusters in a bimodal distribution."""
    # Handle empty tensor case
    if values.numel() == 0:
        return torch.tensor(0.0), torch.tensor(1.0)

    # Handle single value case
    if values.numel() == 1:
        return values[0], values[0] + 1.0

    max_iterations = 20

    # Use dim=0 for min/max on empty dimensions
    if values.numel() == 0:
        return torch.tensor(0.0), torch.tensor(1.0)

    previous_low_center = values.min()
    previous_high_center = values.max()

    # If min and max are the same, slightly modify high_center
    if previous_low_center == previous_high_center:
        previous_high_center = previous_low_center + 1.0

    convergence_threshold = (previous_high_center - previous_low_center) / 100

    for _ in range(max_iterations):
        high_center = average_of_near_values(values, previous_high_center, previous_low_center)
        low_center = average_of_near_values(values, previous_low_center, previous_high_center)

        if (torch.abs(high_center - previous_high_center) < convergence_threshold and
                torch.abs(low_center - previous_low_center) < convergence_threshold):
            return low_center, high_center

        previous_high_center = high_center
        previous_low_center = low_center

    print("Warning: findClusterMeans exceeded maxIterations without converging")
    print("Previous High Center:", previous_high_center.item())
    print("High Center:", high_center.item())
    print("Previous Low Center:", previous_low_center.item())
    print("Low Center:", low_center.item())

    return low_center, high_center



def mfcc(signal, rate, win_length, hop_length, preemph, window_fn, freq_range, num_filters, num_ceps):
    """Compute MFCC features."""
    mfcc_transform = T.MFCC(
        sample_rate=rate,
        n_mfcc=num_ceps,
        melkwargs={
            'n_fft': 400,
            'win_length': int(win_length * rate / 1000),
            'hop_length': int(hop_length * rate / 1000),
            'n_mels': num_filters,
            'f_min': freq_range[0],
            'f_max': freq_range[1],
            'window_fn': window_fn
        }
    )
    return mfcc_transform(signal)


def smooth_jcc(vector, smoothingSize):
    """
    Smooths the points in vector using a rolling average of the
    surrounding smoothingSize points, except for the
    first floor(smoothingSize/2) points and the last
    floor(smoothingSize/2) points. 
    Jason Carlson's reimplementation of Matlab's smooth() function.
    UTEP, January 2017
    """
    oddvlen = vector.nelement()
    if oddvlen % 2 == 0:
        oddvlen = oddvlen - 1
    
    smoothingSize = min(smoothingSize, oddvlen)
    smoothed = torch.zeros_like(vector)
    
    csum = torch.cumsum(vector, 0)
       
    index = 0
    for i in range(0, smoothingSize, 2):
        smoothed[index] = csum[i]/(i+1)       
        index = index + 1
    
    start = index
    finish = vector.nelement() - math.floor(smoothingSize/2)
  
    # integral vector for speed
    if start <= finish:
        smoothed[start:finish] = (csum[smoothingSize:] - csum[:(vector.nelement() - smoothingSize)])/smoothingSize
    
    index = finish
    for i in range(vector.nelement() - smoothingSize + 1, vector.nelement()-1, 2):
        smoothed[index] = (csum[-1] - csum[i])/(vector.nelement()-(i+1));
        index = index + 1
    return smoothed


def cepstral_flux(signal, rate, energy):
    """Compute the cepstral flux from a given signal."""
    cc = mfcc(signal, rate, 25, 10, 0.97, torch.hamming_window, [300, 3700], 20, 13)
    cc = torch.cat((torch.zeros(13, 1), cc, torch.zeros(13, 1)), dim=1)
    
    smoothing_size = 15  # smooth over 150ms windows 
    cct = cc.T
    diff = cct[1:] - cct[:-1]
    
    if diff.shape[0] < len(energy):
        avg_diff = torch.mean(torch.abs(diff), dim=0, keepdim=True)
        diff = torch.cat((avg_diff, diff), dim=0)
    
    diff_squared = diff ** 2
    sum_diff_sq = torch.sum(diff_squared, dim=1)
    
    smoothed = smooth_jcc(sum_diff_sq, smoothing_size)
    return smoothed


def windowize(frameFeatures, msPerWindow):
    """
    % inputs:
    %   frameFeatures: features over every 10 millisecond frame,
    %     centered at 5ms, 15ms etc. 
    %     A row vector.
    %   msPerWindow: duration of window over which to compute windowed values
    % output:
    %   summed values over windows of the designated size, 
    %     centered at 10ms, 20ms, etc.
    %     (the centering is off, at 15ms, etc, if msPerWindow is 30ms, 50ms etc)
    %      but we're not doing syllable-level prosody, so it doesn't matter.
    %   values are zero if either end of the window would go outside  
    %     what we have data for.

    %% There are other ways to windowize (mean, std-dev, range, etc),
    %%  so this should probably be called windowizeSum 

    % Nigel Ward, UTEP, Feb 2015
    """

    integralImage = torch.cat([torch.tensor([0]), torch.cumsum(frameFeatures, 0)])
    framesPerWindow = msPerWindow // 10
    windowSum = integralImage[framesPerWindow:] - integralImage[0:-framesPerWindow]

    # align so first value is for window centered at 10 ms 
    #  (or 15ms if, an odd number of frames)
    headFramesToPad = math.floor(framesPerWindow / 2) - 1
    tailFramesToPad = math.ceil(framesPerWindow / 2);
    windowValues = torch.cat([torch.zeros(headFramesToPad), windowSum, torch.zeros(tailFramesToPad)]);
    return windowValues

if __name__ == "__main__":
#    smooth_jcc(torch.arange(105), 15)
#    misalignment(torch.arange(1, 101), torch.arange(5, 105))
#    filter_kernel = rectangular_filter(123)

#    print(windowize(torch.tensor([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 20))
#    print(windowize(torch.tensor([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 30))
    a, b = find_cluster_means(torch.tensor([1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 4, 6, 7, 6, 7, 6, 7, 6, 7, 1, 9, 0, 6, 6, 3], dtype=float))
    import ipdb ; ipdb.set_trace()