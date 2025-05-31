import pandas as pd
import numpy as np
import os
import soundfile as sf
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(
            self,
            data_path,
            files,
            load_fn=sf.read,
            return_utt_id=False,
            subset_num=-1,
            # batch_length=9600,
            # batch_length=240000  # 16000 * 15 = 15 seconds of each audio file
            batch_length=80000  # 16000 * 5 = 5 seconds of each audio file
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.subset_num = subset_num
        self.data_path = data_path
        self.batch_length = batch_length

        # Load and process paths from CSV
        self.filenames = self._load_list(files)

        # Print the first filename
        # if len(self.filenames) > 0:
        #     print(f"FFFFFFFFFFFFFFFirst filename: {self.filenames[0]}")
        # else:
        #     print("Nooooooooooooo filenames found.")

        if subset_num > 0:
            self.filenames = self.filenames[:subset_num]

        # Generate utterance IDs
        self.utt_ids = self._load_ids(self.filenames)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        data, data_shape = self._data(idx)
        filename = self.filenames[idx]  # Azam: Added this line to get the filename

        if self.return_utt_id:
            items = (utt_id, (data, data_shape), filename)
        else:
            items = (data, data_shape, filename)

        return items

    def __len__(self):
        return len(self.filenames)

    def _load_list(self, files):
        if isinstance(files, str):
            # If it's a CSV file
            df = pd.read_csv(files)
            if 'path' in df.columns:
                filenames = df['path'].tolist()
            else:
                raise ValueError("CSV file must contain a 'path' column")
        else:
            # If it's already a list of paths
            filenames = files

        # Ensure all paths are absolute
        filenames = [self._ensure_absolute_path(f) for f in filenames]
        return filenames


    def _ensure_absolute_path(self, filepath):
        """Convert relative paths to absolute paths."""
        # Azam: commented the 2 below lines to not recognize path with "/" as an absolute path
        # if os.path.isabs(filepath):
        #     return filepath
        # Remove leading '/' if present to avoid double slashes
        filepath = filepath.lstrip('/')
        return os.path.join(self.data_path, filepath)

    def _load_ids(self, filenames):
        """Generate utterance IDs from filenames."""
        return [os.path.splitext(os.path.basename(f))[0] for f in filenames]

    def _data(self, idx):
        """Load data for a given index."""
        return self._load_data(self.filenames[idx], self.load_fn)

    def _load_data(self, filename, load_fn):
        if load_fn == sf.read:
            # Load audio data
            data = load_fn(filename, always_2d=True)[0][:, 0]  # T x C, 1
            data_shape = data.shape[0]

            # print(f"Filename: {filename}")
            # print(f"Original data shape (number of samples): {data_shape}")

            # Process audio data
            if data_shape <= self.batch_length:
                # Pad if audio is shorter than batch length
                padding = np.zeros(self.batch_length - data_shape)
                data = np.concatenate((data, padding))[None, :].astype(np.float32)
            else:
                # # Random crop if audio is longer than batch length
                # start = np.random.randint(0, data_shape - self.batch_length + 1)
                # Azam: Use deterministic crop instead of random crop
                start = self.deterministic_start_index(filename, data_shape)
                data = data[None, start: start + self.batch_length].astype(np.float32)
        else:
            # Handle other file types
            data = load_fn(filename)
            data_shape = data.shape[0] if hasattr(data, 'shape') else len(data)

        return data, data_shape

    def deterministic_start_index(self, filename, data_shape):
        # Use a hash of the filename to generate a reproducible "random" index
        import hashlib

        # Get a hash of the filename
        filename_hash = hashlib.md5(filename.encode()).hexdigest()

        # Convert first 8 characters of hash to integer
        hash_int = int(filename_hash[:8], 16)

        # Scale to get an index within the valid range
        if data_shape <= self.batch_length:
            return 0  # No need for selection if audio is shorter
        else:
            # Get a value in range [0, audio_length - target_length]
            valid_range = data_shape - self.batch_length
            return hash_int % (valid_range + 1)
