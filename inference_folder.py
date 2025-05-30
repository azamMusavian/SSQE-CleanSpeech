# -*- coding: utf-8 -*-
"""
@author: szuweif
"""
import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
import pandas as pd
import time
from vqscore.models.vqvae_encoder_decoder import VQVAE_QE  # New model with BiLSTM

def get_filepaths(directory):
    file_paths = []  # List to store all the full file paths.
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.wav')):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
    return file_paths


def resynthesize(enhanced_mag, noisy_inputs, hop_size):
    """Resynthesize waveforms from enhanced magnitudes."""
    noisy_feats = torch.stft(
        noisy_inputs,
        n_fft=512,
        hop_length=hop_size,
        win_length=512,
        window=torch.hamming_window(512).to(device),
        center=True,
        pad_mode="constant",
        onesided=True,
        return_complex=False
    ).transpose(2, 1)

    noisy_phase = torch.atan2(
        noisy_feats[:, :, :, 1],
        noisy_feats[:, :, :, 0]
    )[:, 0:enhanced_mag.shape[1], :]
    predictions = torch.mul(
        torch.unsqueeze(enhanced_mag, -1),
        torch.cat(
            (
                torch.unsqueeze(torch.cos(noisy_phase), -1),
                torch.unsqueeze(torch.sin(noisy_phase), -1),
            ),
            -1,
        ),
    ).permute(0, 2, 1, 3)

    complex_predictions = torch.complex(predictions[..., 0], predictions[..., 1])
    pred_wavs = torch.istft(
        input=complex_predictions,
        n_fft=512,
        hop_length=hop_size,
        win_length=512,
        window=torch.hamming_window(512).to(device),
        center=True,
        onesided=True,
        length=noisy_inputs.shape[1]
    )
    return pred_wavs


def stft_magnitude(x, hop_size, fft_size=512, win_length=512):
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window=torch.hann_window(win_length).to(device),
        return_complex=False
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # Compute magnitude and transpose to shape [B, T, F]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def cos_loss(SP_noisy, SP_y_noisy):
    eps = 1e-5
    SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True) + eps
    SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True) + eps
    Cos_frame = torch.sum(
        (SP_noisy / SP_noisy_norm) * (SP_y_noisy / SP_y_noisy_norm),
        dim=-1
    )
    return -torch.mean(Cos_frame)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-m', '--path_of_model_weights', type=str, required=True)
parser.add_argument('-i', '--path_of_input_audio_folder', type=str, required=True)
parser.add_argument('-o', '--path_of_output_audio_folder', type=str, default="./enhanced/")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

with torch.no_grad():
    if config['task'] == "Quality_Estimation":
        hop_size = 256
        # Instantiate the VQVAE model with the new BiLSTM+CNN structure.
        VQVAE = VQVAE_QE(**config['VQVAE_params']).to(device).eval()
        VQVAE.load_state_dict(
            torch.load(args.path_of_model_weights, map_location=device)['model']['VQVAE']
        )

        file_list = get_filepaths(args.path_of_input_audio_folder)
        original_VQ = []
        start_time = time.time()

        for file in file_list:
            speech, fs = torchaudio.load(file)
            if fs != 16000:
                speech = torchaudio.functional.resample(speech, fs, 16000).to(device)

            # Compute STFT magnitude features.
            SP_original = stft_magnitude(speech, hop_size=hop_size)
            if config['input_transform'] == 'log1p':
                SP_original = torch.log1p(SP_original)

            print('file path for prosodic features', file)
            out, zq, indices, vqloss, distance, z = VQVAE(SP_original.to(device), stochastic=False, update=False, audio_path=file)

            # Compute VQScore (cosine loss between the combined latent representation and its quantized version).
            VQScore_cos = -cos_loss(z.cpu(), zq.cpu()).numpy()
            original_VQ.append(VQScore_cos)

        sort_index = np.argsort(original_VQ)
        sorted_VQ = [original_VQ[i] for i in sort_index]
        sorted_file_list = [file_list[i] for i in sort_index]
        score_dict = {'filename': sorted_file_list, 'VQScore': sorted_VQ}

        df = pd.DataFrame.from_dict(score_dict)
        df.to_csv('VQscore.csv', index=False)
        end_time = time.time()

        print('Total number of files evaluated:', len(original_VQ))
        print('Average VQScore:', np.mean(original_VQ))
        print('VQScore list (in ascending order) has been saved in the ./VQscore.csv')
        print('The evaluation takes around %.2fmin' % ((end_time - start_time) / 60.))
