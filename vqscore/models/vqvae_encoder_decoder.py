import librosa
import torch
import numpy as np
import soundfile as sf
import os
import pandas as pd
import torch.nn.functional as F
from .vqvae_discrete_latent_converter import VectorQuantize


# -----------------------------
# 1. CNN Encoder (using InstanceNorm1d)
# -----------------------------
class CNN_1D_encoder_QE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)

        # Normailization layer
        self.enc_In0 = torch.nn.InstanceNorm1d(257)
        self.enc_In1 = torch.nn.InstanceNorm1d(128)
        self.enc_In2 = torch.nn.InstanceNorm1d(128)
        self.enc_In3 = torch.nn.InstanceNorm1d(64)
        self.enc_In4 = torch.nn.InstanceNorm1d(64)
        self.enc_In5 = torch.nn.InstanceNorm1d(codebook_dim)
        self.enc_In6 = torch.nn.InstanceNorm1d(codebook_dim)
        self.enc_In7 = torch.nn.InstanceNorm1d(codebook_dim)

        ## Encoder
        self.conv_enc1 = torch.nn.Conv1d(in_channels=257, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_enc2 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_enc3 = torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_enc4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_enc5 = torch.nn.Conv1d(in_channels=64, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        self.conv_enc6 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1,
                                         padding=3)

    def forward(self, x):  # x.shape = torch.Size([B, T, 257])
        x = self.enc_In0(x.transpose(2, 1))  # x.shape = torch.Size([B, 257, T])

        enc1 = self.enc_In1(self.activation(self.conv_enc1(x)))  # torch.Size([B, 128, T])
        enc2 = self.enc_In2(self.activation(self.conv_enc2(enc1)))  # torch.Size([B, 128, T])
        enc3 = self.enc_In3(self.activation(self.conv_enc3(enc1 + enc2)))  # torch.Size([B, 64, T])
        enc4 = self.enc_In4(self.activation(self.conv_enc4(enc3)))  # torch.Size([B, 64, T])
        enc5 = self.enc_In5(self.activation(self.conv_enc5(enc3 + enc4)))  # torch.Size([B, 32, T])
        z = self.enc_In6(self.conv_enc6(enc5))  # torch.Size([B, 32, T])
        return z


class CNN_1D_decoder_QE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)

        self.conv_dec1 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1,
                                         padding=3)
        self.conv_dec2 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_dec3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_dec4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_dec5 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_dec6 = torch.nn.Conv1d(in_channels=128, out_channels=257, kernel_size=7, stride=1, padding=3)

    def forward(self, zq):
        dec1 = (self.activation(self.conv_dec1(zq.transpose(2, 1))))
        dec2 = (self.activation(self.conv_dec2(dec1)))
        dec3 = (self.activation(self.conv_dec3(dec2)))
        dec4 = (self.activation(self.conv_dec4(dec3 + dec2)))
        dec5 = (self.activation(self.conv_dec5(dec4)))
        out = F.relu(self.conv_dec6(dec5 + dec4).transpose(2, 1))  # torch.Size([B, T, 257])
        return out


class CNN_1D_quantizer_QE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update,
                 learnable_codebook,
                 stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init,
                 threshold_ema_dead_code,
                 ):
        super().__init__()

        self.quantizer = VectorQuantize(
            dim=codebook_dim,
            codebook_size=codebook_size,
            use_cosine_sim=use_cosine_sim,
            orthogonal_reg_weight=orthogonal_reg_weight,  # in paper, they recommended a value of 10
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1,  # the weight on the commitment loss
            kmeans_init=kmeans_init,  # set to True
            kmeans_iters=10,  # number of kmeans iterations to calculate the centroids for the codebook on init
            heads=codebook_num,
            separate_codebook_per_head=True,
            ema_update=ema_update,
            learnable_codebook=learnable_codebook,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
            straight_through=straight_through,
            reinmax=reinmax,
            threshold_ema_dead_code=threshold_ema_dead_code
        )

    def forward(self, z, stochastic, update=True, indices=None):  # x.shape = torch.Size([B, T, 257])
        if indices == None:
            zq, indices, vqloss, distance = self.quantizer(z, stochastic, update=update)
            return zq, indices, vqloss, distance
        else:
            zq, cross_entropy_loss = self.quantizer(z, stochastic, indices=indices)
            return zq, cross_entropy_loss


# -----------------------------
# Prosodic Feature Processor
# -----------------------------
class ProsodicsProcessor(torch.nn.Module):
    def __init__(self, prosodic_dim=36, output_dim=32):
        super().__init__()

        # Add standardization for prosodic features
        self.prosodic_processor = torch.nn.Sequential(
            # torch.nn.LayerNorm(prosodic_dim),
            torch.nn.Linear(prosodic_dim, output_dim),
            torch.nn.LeakyReLU(negative_slope=0.3),
            # torch.nn.LayerNorm(output_dim),
            torch.nn.Dropout(0.1)  # Add dropout for better generalization
        )

    def forward(self, prosodic_features):
        return self.prosodic_processor(prosodic_features)

# -----------------------------
# Enhanced Feature Reduction Module
# -----------------------------
class EnhancedFeatureReduction(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # First path: direct projection with attention to features
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.LeakyReLU(negative_slope=0.3),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.Sigmoid()
        )

        # Main projection path with non-linearity
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.LeakyReLU(negative_slope=0.3),
            torch.nn.LayerNorm(in_dim // 2),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_dim // 2, out_dim),
            torch.nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # Apply feature attention
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights

        # Apply projection
        return self.projection(weighted_features)


# -----------------------------
# 4. VQVAE Model with Enhanced Feature Reduction
# -----------------------------
class VQVAE_QE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update,
                 learnable_codebook, stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax,
                 kmeans_init, threshold_ema_dead_code, use_prosodic_features=True):
        super().__init__()

        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(code_dir, os.pardir, os.pardir))
        self.prosody_root = os.path.join(project_root, "data", "metadata", "prosody")

        self.prosodic_norm = None
        self.batch_length = 80000
        self.use_prosodic_features = use_prosodic_features
        self.prosodic_weight = torch.nn.Parameter(torch.tensor(0.3))  # Learnable weight

        self.CNN_1D_encoder = CNN_1D_encoder_QE(codebook_dim)
        # Add a bidirectional LSTM layer. Using hidden_size=codebook_dim//2 so that
        # the concatenated output (forward + backward) remains codebook_dim.
        self.bilstm = torch.nn.LSTM(input_size=codebook_dim,
                                    hidden_size=codebook_dim // 2,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        self.lstm_norm = torch.nn.LayerNorm(codebook_dim)
        self.feature_reduction = EnhancedFeatureReduction(2 * codebook_dim, codebook_dim)

        self.spectral_quantizer = CNN_1D_quantizer_QE(
            codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight,
            use_cosine_sim, ema_update, learnable_codebook, stochastic_sample_codes,
            sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code
        )

        # Prosodic path components
        if use_prosodic_features:
            self.load_global_prosodic_stats()
            self.prosodic_processor = ProsodicsProcessor(prosodic_dim=36, output_dim=codebook_dim)
            self.prosodic_quantizer = CNN_1D_quantizer_QE(
                codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight,
                use_cosine_sim, ema_update, learnable_codebook, stochastic_sample_codes,
                sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code
            )

        # Decoder uses only spectral features
        self.CNN_1D_decoder = CNN_1D_decoder_QE(codebook_dim)


    def forward(self, x, stochastic, audio_paths=None, update=True, indices=None):
        # x shape: [B, T, 257]= [64,313,257] # a hop size of ~256 would give ~313 time steps (16ms) for 5 seconds of audio.
        # CNN encoder output: shape [B, T, codebook_dim]
        z = self.CNN_1D_encoder(x).transpose(1, 2)
        z_lstm, _ = self.bilstm(z)
        z_lstm = self.lstm_norm(z_lstm)
        z_combined = torch.cat([z, z_lstm], dim=-1)  # shape: [B, T, 2*codebook_dim]
        z_spectral_features = self.feature_reduction(z_combined)

        if self.use_prosodic_features and audio_paths is not None:
            batch_pros = []
            for path in audio_paths:
                # 1) Build the CSV filename and load it as a DataFrame
                audio_fn = os.path.basename(path)  # e.g. "2598-4654-0035.wav"
                pros_fn = audio_fn.replace(".wav", "_prosody.csv")  # → "2598-4654-0035_prosody.csv"
                csv_p = os.path.join(self.prosody_root, pros_fn)
                df = pd.read_csv(csv_p)

                # 2) Apply global normalization if available
                if self.prosodic_norm is not None:
                    # Get feature columns in the same order as we computed statistics
                    feature_cols = self.prosodic_norm['feature_columns']
                    # Extract features in the correct order
                    if all(col in df.columns for col in feature_cols):
                        features = df[feature_cols].values.astype(np.float32)
                        # Apply global normalization
                        global_mean = self.prosodic_norm['mean'].to(x.device)
                        global_std = self.prosodic_norm['std'].to(x.device)
                        normed = (features - global_mean.cpu().numpy()) / global_std.cpu().numpy()
                    else:
                        # Fallback to per-file normalization if columns don't match
                        print(
                            f"Warning: Columns in {pros_fn} don't match expected feature columns. Using per-file normalization.")
                        features = df.drop(["filename", "time_frame"], axis=1).values.astype(np.float32)
                        mu = features.mean(axis=0, keepdims=True)
                        sigma = features.std(axis=0, keepdims=True)
                        sigma[sigma < 1e-5] = 1.0
                        normed = (features - mu) / sigma
                else:
                    # Fallback to per-file normalization
                    features = df.drop(["filename", "time_frame"], axis=1).values.astype(np.float32)
                    mu = features.mean(axis=0, keepdims=True)
                    sigma = features.std(axis=0, keepdims=True)
                    sigma[sigma < 1e-5] = 1.0
                    normed = (features - mu) / sigma

                # 3) Convert to a Torch tensor on the correct device
                pros = torch.from_numpy(normed).to(x.device)
                batch_pros.append(pros)

            # now stack and run through your prosodic_processor exactly as before:
            pros_raw = torch.stack(batch_pros, dim=0)  # [B, T, P]
            B, T, P = pros_raw.shape

            pros_flat = pros_raw.view(B * T, P)  # [B*T, P]
            pros_proc_flat = self.prosodic_processor(pros_flat)  # [B*T, Dp]
            Dp = pros_proc_flat.size(-1)
            pros_processed = pros_proc_flat.view(B, T, Dp)  # [B, T, Dp]

            T_spec = z_spectral_features.size(1)
            T_pros = pros_processed.size(1)
            T_min = min(T_spec, T_pros)
            z_spec_trunc = z_spectral_features[:, :T_min, :]  # [B, T_min, Ds]
            pros_trunc = pros_processed[:, :T_min, :]  # [B, T_min, Dp]

            z_final = torch.cat([z_spec_trunc, pros_trunc], dim=-1)

        if indices is None:
            prosodic_zq, prosodic_indices, prosodic_vqloss, prosodic_distance = self.prosodic_quantizer(
                pros_processed, stochastic, update)
            spectral_zq, spectral_indices, spectral_vqloss, spectral_distance = self.spectral_quantizer(
                z_spectral_features, stochastic, update)

            T_spec = spectral_zq.size(1)
            T_pros = prosodic_zq.size(1)
            T_min = min(T_spec, T_pros)
            z_spec_trunc = spectral_zq[:, :T_min, :]  # [B, T_min, Ds]
            pros_trunc = prosodic_zq[:, :T_min, :]  # [B, T_min, Dp]

            zq = torch.cat([pros_trunc, z_spec_trunc], dim=-1)
            indices_combined = (spectral_indices, prosodic_indices)

            # Combine losses with weighting
            vqloss = spectral_vqloss + self.prosodic_weight * prosodic_vqloss

            T_min = min(spectral_distance.size(2), prosodic_distance.size(2))
            spectral_distance = spectral_distance[:, :, :T_min, :]
            prosodic_distance = prosodic_distance[:, :, :T_min, :]
            distance = spectral_distance + self.prosodic_weight * prosodic_distance
            # --- Decode ---
            out = self.CNN_1D_decoder(spectral_zq)
            return out, zq, indices_combined, vqloss, distance, z_final

        else:

            spectral_indices, prosodic_indices = indices if isinstance(indices, tuple) else (indices, indices)
            prosodic_zq, prosodic_ce_loss = self.prosodic_quantizer(pros_processed, stochastic, indices=prosodic_indices)
            spectral_zq, spectral_ce_loss = self.spectral_quantizer(z_spectral_features, stochastic, indices=spectral_indices)

            T_spec = spectral_zq.size(1)
            T_pros = prosodic_zq.size(1)
            T_min = min(T_spec, T_pros)
            z_spec_trunc = spectral_zq[:, :T_min, :]  # [B, T_min, Ds]
            pros_trunc = prosodic_zq[:, :T_min, :]  # [B, T_min, Dp]

            zq = torch.cat([pros_trunc, z_spec_trunc], dim=-1)

            # Combine the losses with weighting
            cross_entropy_loss = spectral_ce_loss + (self.prosodic_weight * prosodic_ce_loss)

            # --- Decode ---
            out = self.CNN_1D_decoder(spectral_zq)
            return out, zq, cross_entropy_loss, z_final

    def load_global_prosodic_stats(self):
        """
        Computes global mean and std for prosodic features across all files
        with the format '986-129388-0112_prosody.csv', handling each feature column separately.
        """
        if self.prosodic_norm is not None:
            return  # Already computed

        # Get list of all prosodic CSV files that match the correct format
        import re
        pattern = r'^\d+-\d+-\d+_prosody\.csv$'  # Pattern like '986-129388-0112_prosody.csv'

        prosody_files = [f for f in os.listdir(self.prosody_root)
                         if f.endswith("_prosody.csv") and re.match(pattern, f)]

        print(f"Computing global prosodic statistics from {len(prosody_files)} training files...")

        if len(prosody_files) == 0:
            print("No matching prosodic data files found. Using per-file normalization.")
            return

        # Sample a subset of files if there are too many
        max_files = 1000  # Adjust based on memory constraints
        if len(prosody_files) > max_files:
            import random
            random.shuffle(prosody_files)
            prosody_files = prosody_files[:max_files]

        # First, identify all feature columns by reading the first file
        first_file = os.path.join(self.prosody_root, prosody_files[0])
        try:
            first_df = pd.read_csv(first_file)
            feature_columns = [col for col in first_df.columns if col not in ["filename", "time_frame"]]
        except Exception as e:
            print(f"Error reading first file {prosody_files[0]}: {e}")
            return

        # Initialize storage for computing running stats for each feature
        feature_values = {col: [] for col in feature_columns}

        # Collect data for each feature column separately
        for pros_fn in prosody_files:
            csv_p = os.path.join(self.prosody_root, pros_fn)
            try:
                df = pd.read_csv(csv_p)
                # For each feature column, collect values
                for col in feature_columns:
                    if col in df.columns:
                        feature_values[col].append(df[col].values)
            except Exception as e:
                print(f"Error loading {pros_fn}: {e}")
                continue

        # Compute mean and std for each feature column separately
        global_mean = np.zeros((1, len(feature_columns)), dtype=np.float32)
        global_std = np.zeros((1, len(feature_columns)), dtype=np.float32)

        for i, col in enumerate(feature_columns):
            if feature_values[col]:
                # Concatenate all values for this feature
                all_values = np.concatenate(feature_values[col])
                # Compute mean and std
                global_mean[0, i] = np.mean(all_values)
                global_std[0, i] = np.std(all_values)
                # Prevent division by zero
                if global_std[0, i] < 1e-5:
                    global_std[0, i] = 1.0

        # Store as tensors for future use
        self.prosodic_norm = {
            'mean': torch.from_numpy(global_mean),
            'std': torch.from_numpy(global_std),
            'feature_columns': feature_columns
        }
        print("Global prosodic statistics computed successfully.")
        print(f"Feature columns: {feature_columns}")
        print(f"Global means: {global_mean}")
        print(f"Global stds: {global_std}")

