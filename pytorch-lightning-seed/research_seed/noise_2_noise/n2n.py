"""
This file defines the core research contribution
"""
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl
from research_seed.common import *
from research_seed.utils import *
from research_seed.noise_2_noise.dataset import *
from research_seed.noise_2_noise.model_seq import *


class Noise2Noise(pl.LightningModule):
    def __init__(self, hparams):
        super(Noise2Noise, self).__init__()
        self.hparams = hparams
        self.source_noise_model = get_noise_model(hparams.source_noise_model)
        self.target_noise_model = get_noise_model(hparams.target_noise_model)
        self.val_noise_model = get_noise_model(hparams.val_noise_model)
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.loss_type = hparams.loss_type
        # self.data_root = "data/sub_data"
        self.data_root = "data/train"
        # self.data_root = Path(
        # "C:\\Users\\zhang\\Desktop\\Code\\inpaint\\pytorch-lightning-seed\\data\\sub_data"
        # )
        self.saved_output = None
        self.predict_output = None
        self.show_masked_once = False

        if hparams.model_name is "unet":
            self.model = UNet()
        elif hparams.model_name is "srrnet":
            self.model = UNet()
        else:
            raise RuntimeError("Unkown model name found {}".format(hparams.model_name))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        noise, origin, mask, context_mask = batch
        predicted = self.forward(noise)
        self.predict_output = predicted.detach().cpu().numpy().squeeze()
        self.saved_output = (
            predicted.detach().cpu().numpy().squeeze()
            * (1 - mask.detach().cpu().numpy().squeeze())
            + origin.detach().cpu().numpy().squeeze()
            * mask.detach().cpu().numpy().squeeze()
        )
        if not self.show_masked_once:
            self.logger.experiment.add_image(
                f"masked_images",
                origin.detach().cpu().numpy().squeeze()
                * mask.detach().cpu().numpy().squeeze(),
                self.current_epoch,
            )
            self.show_masked_once = True
        if self.loss_type == "mse":
            loss_func = F.mse_loss
        elif self.loss_type == "psnr":
            loss_func = psnr_loss
        elif self.loss_type == "weighted_mse":
            loss_func = weighted_mse_loss
        elif self.loss_type == "weighted_psnr":
            loss_func = weighted_psnr_loss
        else:
            raise RuntimeError("Unknown loss type : {}".format(self.loss_type))
        if "weighted" in self.loss_type:
            loss = loss_func(predicted * mask, origin * mask, context_mask)
        else:
            loss = loss_func(predicted * mask, origin * mask)
        tqdm_dict = {"loss": loss}
        return {"loss": loss, "log": tqdm_dict}

    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.logger.experiment.add_image(
                f"generated_images", self.saved_output, self.current_epoch
            )
            self.logger.experiment.add_image(
                f"predict_images", self.predict_output, self.current_epoch
            )
            if not os.path.exists(self.hparams.ckpt_path):
                os.makedirs(self.hparams.ckpt_path)
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.ckpt_path, "epoch_{}".format(self.current_epoch)
                )
            )

    # def validation_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'val_loss': F.cross_entropy(y_hat, y)}

    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(NoiseImageDataset(self.data_root, source_noise_model, target_noise_model), batch_size=1)

    # @pl.data_loader
    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=1)

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("-t", "--train-dir", help="training data diretory")

        parser.add_argument(
            "--source_noise_model",
            type=str,
            default="text,0,50",
            help="noise model for source images",
        )
        parser.add_argument(
            "--target_noise_model",
            type=str,
            default="text,0,50",
            help="noise model for target images",
        )
        parser.add_argument(
            "--val_noise_model",
            type=str,
            default="text,25,25",
            help="noise model for validation source images",
        )

        parser.add_argument(
            "--image_size", type=int, default=64, help="training patch size"
        )
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")

        parser.add_argument(
            "-n", "--noise-type", help="noise type", default="text", type=str
        )
        parser.add_argument("--learning_rate", default=0.01, type=float)
        # parser.add_argument("--loss_type", default="psnr", type=str)
        # parser.add_argument("--loss_type", default="mse", type=str)
        # parser.add_argument("--loss_type", default="weighted_psnr", type=str)
        parser.add_argument("--loss_type", default="weighted_mse", type=str)
        parser.add_argument("--model_name", default="srrnet", type=str)
        parser.add_argument("--use_gated_conv", action="store_true")

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=5000, type=int)

        return parser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = Noise2Noise.add_model_specific_args(parser)

    hparams = parser.parse_args()

    Noise2Noise(hparams)
