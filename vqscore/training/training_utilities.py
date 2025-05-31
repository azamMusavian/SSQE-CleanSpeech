import os
import sys
import yaml
import random
import logging
import torch
import numpy as np

class Train:
    def __init__(self, args):
        self._setup_logging(args)
        self._set_seed(args)
        self._load_config(args)
        self._create_output_dir(args)

        self.data_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        self.resume = args.resume

    def _setup_logging(self, args):
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    def _set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            if args.disable_cudnn == "False":
                torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"device: {self.device}")

    def _load_config(self, args):
        with open(args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config.update(vars(args))

    def _create_output_dir(self, args):
        self.expdir = os.path.join(args.exp_root, args.tag)
        os.makedirs(self.expdir, exist_ok=True)
        self.config["outdir"] = self.expdir
        with open(os.path.join(self.expdir, "config.yml"), "w") as f:
            yaml.dump(self.config, f, Dumper=yaml.Dumper)
        for key, value in self.config.items():
            logging.info(f"[Train] {key} = {value}")

    def initialize_data_loader(self):
        pass

    def define_model(self):
        pass

    def define_trainer(self):
        pass

    def initialize_model(self):
        if self.resume:
            self.trainer.load_checkpoint(self.resume)
            logging.info(f"Successfully resumed from {self.resume}.")
        elif self.config.get("initial", ""):
            self.trainer.load_checkpoint(self.config["initial"], load_only_params=True)
            logging.info(f"Successfully initialize parameters from {self.config['initial']}.")
        else:
            logging.info("Train from scratch")

    def run(self):
        try:
            self.trainer.run()
        finally:
            self.trainer.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.trainer.steps}steps.")

    def _define_optimizer_scheduler(self):
        # Create optimizer
        optimizer_class = getattr(torch.optim, self.config['VQVAE_optimizer_type'])
        optimizer = optimizer_class(
            self.model['VQVAE'].parameters(),
            **self.config['VQVAE_optimizer_params']
        )
        self.optimizer = {'VQVAE': optimizer}  # Keep dictionary structure

        # Create scheduler
        scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.get('VQVAE_scheduler_type', 'StepLR')
        )
        scheduler = scheduler_class(
            optimizer=self.optimizer['VQVAE'],
            **self.config['VQVAE_scheduler_params']
        )
        self.scheduler = {'VQVAE': scheduler}  # Keep dictionary structure

    def _show_setting(self):
        logging.info(self.model['VQVAE'])
        logging.info(self.optimizer['VQVAE'])
        logging.info(self.scheduler['VQVAE'])