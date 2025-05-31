import os
import logging
import argparse
import pandas as pd
import soundfile as sf
from torch.utils.data import DataLoader

from vqscore.dataloader.wav_audio_batch_processor import SingleDataset
from vqscore.models.vqvae_encoder_decoder import VQVAE_QE
from vqscore.training.vqvae_training_and_plotting import Trainer
from vqscore.training.training_utilities import Train


class TrainMain(Train):
    """
    Main training class for the VQ-VAE autoencoder model.
    Handles the training pipeline including data loading,
    model initialization, and training execution.
    """

    def __init__(self, args):
        super(TrainMain, self).__init__(args=args)
        self.data_path = self.config['data']['path']

    def initialize_data_loader(self):
        logging.info("Loading dataset...")

        train_set = self._audio('clean_train')
        valid_set = self._audio('clean_valid')

        # Total Number of Speakers of train-clean-460:
        # 251 (train-clean-100) + 921 (train-clean-360) = 1,172 Speakers
        # Total Number of Audio Files:
        # 28,539 (train-clean-100) + 104,014 (train-clean-360) = 132,553 Audio Files
        logging.info(f"The number of training files = {len(train_set)}.")  # 132553
        logging.info(f"The number of validation files = {len(valid_set)}.")  # 824
        dataset = {'train': train_set, 'dev': valid_set}
        self._data_loader(dataset)

    def _audio(self, subset, subset_num=-1, return_utt_id=False):
        audio_dir = os.path.join(self.data_path, self.config['data']['subset'][subset])
        print(f"Audio directory: {audio_dir}")  # Add this line to print audio_dir
        params = {
            'data_path': '/',
            'files': audio_dir,
            'query': "*.wav",
            'load_fn': sf.read,
            'return_utt_id': return_utt_id,
            'subset_num': subset_num,
            'batch_length': self.config['batch_length'],
        }
        return SingleDataset(**params)

    def _audio(self, subset, subset_num=-1, return_utt_id=False):
        csv_path = os.path.join(self.data_path, self.config['data']['subset'][subset])
        print(f"csv_path:::::::::::::::::: {csv_path}")
        # Handle relative paths in config
        if csv_path.startswith('./'):
            csv_path = os.path.join(self.data_path, csv_path[2:])

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        logging.info(f"Loading {subset} data from: {csv_path}")

        # Load and process CSV
        try:
            df = pd.read_csv(csv_path)
            file_paths = df['path'].tolist()

            if subset_num > 0:
                file_paths = file_paths[:subset_num]

            logging.info(f"Loaded {len(file_paths)} files from {csv_path}")
            logging.info(f"AAAAbsolute path {self.data_path}")

            params = {
                'data_path': self.data_path,
                'files': file_paths,
                'load_fn': sf.read,
                'return_utt_id': return_utt_id,
                'batch_length': self.config['batch_length'],
            }
            return SingleDataset(**params)

        except Exception as e:
            logging.error(f"Error loading CSV file {csv_path}: {str(e)}")
            raise

    def define_model(self):
        """Initialize the VQ-VAE model for quality estimation."""
        VQVAE = VQVAE_QE(**self.config['VQVAE_params']).to(self.device)
        self.model = {"VQVAE": VQVAE}
        self._define_optimizer_scheduler()

    def define_trainer(self):
        self._show_setting()
        trainer_parameters = {}
        trainer_parameters['steps'] = 0
        trainer_parameters['epochs'] = 0
        trainer_parameters['data_loader'] = self.data_loader
        trainer_parameters['model'] = self.model
        trainer_parameters['criterion'] = self.criterion
        trainer_parameters['optimizer'] = self.optimizer
        trainer_parameters['scheduler'] = self.scheduler
        trainer_parameters['config'] = self.config
        trainer_parameters['device'] = self.device
        self.trainer = Trainer(**trainer_parameters)

    def _data_loader(self, dataset):
        """
        Create DataLoader object for training dataset.
        """
        self.data_loader = {
            'train': DataLoader(
                dataset=dataset['train'],
                shuffle=True,
                collate_fn=None,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
            ),
            'dev': DataLoader(
                dataset=dataset['dev'],
                shuffle=False,
                collate_fn=None,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                drop_last=False
            )
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help='checkpoint file path to resume training. (default="")')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--disable_cudnn', choices=('True', 'False'), default='False',
                        help='Disable CUDNN')
    args = parser.parse_args()

    # Initialize and run the training pipeline
    train_main = TrainMain(args=args)
    train_main.initialize_data_loader()
    train_main.define_model()
    train_main.define_trainer()
    train_main.initialize_model()
    train_main.run()


if __name__ == "__main__":
    main()
