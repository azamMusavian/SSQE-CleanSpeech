import matlab.engine
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch
import hashlib
import soundfile as sf
import librosa
import tempfile

# Define a constant for batch length
BATCH_LENGTH = 80000

class MatlabProsodicExtractor:
    def __init__(self,
                 midlevel_path: str = "/Users/azam/Desktop/midlevel-master",
                 featurelist_mat: str = "new_featurelist.mat",
                 device: torch.device = torch.device("cpu")):
        self.feature_codes = [
            'vo', 'vf', 'sf', 'sr', 'vr',
            'fp', 'np', 'wp', 'th', 'tl', 'pd', 'cr', 'hp', 'lp',
            'cp',
            'le', 'en', 're',
        ]
        self.windows = [(-200, 200), (-50, 50)]
        # 1) Start engine once
        self.eng = matlab.engine.start_matlab()
        # 2) Add paths
        self.eng.addpath(os.path.join(midlevel_path, "src"), nargout=0)
        self.eng.addpath(os.path.join(midlevel_path, "src/voicebox"), nargout=0)
        # 3) Load featureList into MATLAB workspace
        mat_path = os.path.join(midlevel_path, "src", featurelist_mat)
        self.eng.load(mat_path, nargout=0)
        self.device = device
        # Compute project root as two levels up from this file
        this_file = os.path.abspath(__file__)
        self.project_root = "/Users/azam/Desktop/Thesis/speech_quality_assessment"
        self.csv_dir = os.path.join(self.project_root, "data", "metadata", "vctk_clean_prosody")
        os.makedirs(self.csv_dir, exist_ok=True)

    def extract(self, audio_file_path: str, original_path: str) -> torch.Tensor:
        filename = os.path.basename(original_path)
        directory = os.path.dirname(audio_file_path) + "/"

        # build trackspec struct in MATLAB
        ts = self.eng.struct()
        ts["path"] = audio_file_path
        ts["side"] = "l"
        ts["filename"] = filename
        ts["directory"] = directory
        self.eng.workspace["trackspec"] = ts

        # call makeTrackMonster → only want the monster matrix
        raw_feats, monster = self.eng.eval("makeTrackMonster(trackspec, featureList)", nargout=2)

        # convert to numpy, then torch
        np_feats = np.array(monster)  # shape [T, 10]

        # replace any NaN with 0.0
        np_feats = np.nan_to_num(np_feats, nan=0.0)

        cols = [code for code in self.feature_codes for _ in self.windows]
        df = pd.DataFrame(np_feats, columns=cols)
        # add time_frame index and filename
        df.insert(0, "time_frame", df.index)
        df.insert(0, "filename", filename)

        csv_name = os.path.splitext(filename)[0] + "_prosody.csv"
        print("csv_name:", os.path.join(self.csv_dir, csv_name))
        df.to_csv(os.path.join(self.csv_dir, csv_name), index=False)

        # print("NumPy array shape:", np_feats.shape)
        return torch.from_numpy(np_feats).float().to(self.device)

    def close(self):
        self.eng.quit()

def deterministic_start_index(filename, data_shape, batch_length=BATCH_LENGTH):
    filename_hash = hashlib.md5(filename.encode()).hexdigest()
    hash_int = int(filename_hash[:8], 16)

    if data_shape <= batch_length:
        return 0
    else:
        valid_range = data_shape - batch_length
        return hash_int % (valid_range + 1)

def crop_to_5s(path: str, batch_length: int):
    data, sr = sf.read(path, always_2d=True)
    mono = data[:, 0]

    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 2) pad or crop
    if mono.shape[0] <= batch_length:
        pad = np.zeros(batch_length - mono.shape[0], dtype=mono.dtype)
        segment = np.concatenate([mono, pad])
    else:
        start = deterministic_start_index(path, mono.shape[0])
        segment = mono[start: start + batch_length]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, segment, sr, format='WAV', subtype='PCM_16')
    return tmp_path

def extract_and_save_prosody_csv( input_csv: str, audio_root: str):
    file_df = pd.read_csv(input_csv)
    extractor = MatlabProsodicExtractor()

    try:
        for _, row in tqdm(file_df.iterrows(), total=len(file_df), desc="Extracting prosody"):
            rel_path = row["path"].lstrip("/")
            full_path = os.path.join(audio_root, rel_path)
            # print(f"audio_path {full_path}")
            tmp_wav = crop_to_5s(full_path, batch_length=80000)
            extractor.extract(tmp_wav, original_path=full_path)
            if os.path.exists(tmp_wav):  # Safety check
                os.remove(tmp_wav)

    finally:
        extractor.close()
    print(f"Wrote prosodic features for {len(file_df)} files")


if __name__ == "__main__":
    extract_and_save_prosody_csv(
        input_csv="/Users/azam/Desktop/Thesis/speech_quality_assessment/data/metadata/vctk_clean_validation.csv",
        audio_root="/Users/azam/Desktop/Thesis/speech_quality_assessment/data"
    )
