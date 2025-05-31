import argparse
import concurrent.futures
import glob
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import onnxruntime as ort

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=['CPUExecutionProvider'])
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=['CPUExecutionProvider'])

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def process_audio(self, file_path):
        try:
            audio, input_fs = sf.read(file_path)
            if input_fs != SAMPLING_RATE:
                audio = librosa.resample(audio, orig_sr=input_fs, target_sr=SAMPLING_RATE)
            return self(audio, SAMPLING_RATE, is_personalized_MOS=False)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def __call__(self, audio, sampling_rate, is_personalized_MOS, is_normalized=False, is_p808=False):
        if is_normalized:
            audio = audio / abs(audio).max()

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * sampling_rate)

        # Pad audio if necessary
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sampling_rate) - INPUT_LENGTH) + 1
        hop_len_samples = sampling_rate

        predicted_mos = {
            'sig_raw': [], 'bak_raw': [], 'ovr_raw': [],
            'sig': [], 'bak': [], 'ovr': [],
            'p808': [] if is_p808 else None
        }

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            if is_p808:
                p808_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
                p808_mos = self.p808_onnx_sess.run(None, {'input_1': p808_features})[0][0][0]
                predicted_mos['p808'].append(p808_mos)

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, {'input_1': input_features})[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)

            for key, value in zip(['sig_raw', 'bak_raw', 'ovr_raw', 'sig', 'bak', 'ovr'],
                                  [mos_sig_raw, mos_bak_raw, mos_ovr_raw, mos_sig, mos_bak, mos_ovr]):
                predicted_mos[key].append(value)

        result = {
            'len_in_sec': actual_audio_len / sampling_rate,
            'sr': sampling_rate,
            'num_hops': num_hops
        }

        for key in predicted_mos:
            if predicted_mos[key] is not None:
                result[key.upper()] = np.mean(predicted_mos[key])

        return result


def main(args):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_models")
    p808_model_path = os.path.join(base_path, 'model_v8.onnx')
    primary_model_path = os.path.join(base_path, 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    # Get all WAV files recursively
    clips = []
    for root, _, files in os.walk(args.testset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                clips.append(os.path.join(root, file))

    if not clips:
        print(f"No WAV files found in {args.testset_dir}")
        return

    rows = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_clip = {executor.submit(compute_score.process_audio, clip): clip
                          for clip in clips}

        for future in tqdm(concurrent.futures.as_completed(future_to_clip)):
            clip = future_to_clip[future]
            try:
                result = future.result()
                if result is not None:
                    result['file_path'] = clip
                    rows.append(result)
            except Exception as exc:
                print(f'{clip} generated an exception: {exc}')

    if rows:
        df = pd.DataFrame(rows)
        if args.csv_path:
            df.to_csv(args.csv_path, index=False)
        else:
            print(df.describe())
    else:
        print("No results were successfully processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.',
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None,
                        help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true',
                        help='Flag to indicate if personalized MOS score is needed or regular')

    args = parser.parse_args()
    main(args)