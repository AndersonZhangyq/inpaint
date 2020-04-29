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
from research_seed.deep_image_prior.dataset import *


class DIP(nn.Module):
    def __init__(self, hparams):
        super(DIP, self).__init__()
        num_input_channels = 2
        num_output_channels = 3
        num_channels_down = [16, 32, 64, 128, 128][: hparams.depth]
        num_channels_up = [16, 32, 64, 128, 128][: hparams.depth]
        num_channels_skip = [4, 4, 4, 4, 4][: hparams.depth]
        filter_size_down = 3
        filter_size_up = 3
        filter_skip_size = 1
        need_sigmoid = True
        upsample_mode = "nearest"
        act_fun = "LeakyReLU"
        need1x1_up = True
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down)

        last_scale = n_scales - 1

        input_depth = num_input_channels

        model = nn.Sequential()
        model_tmp = model

        for i in range(len(num_channels_down)):
            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add_module(str(len(model_tmp)), Concat(1, skip, deeper))
            else:
                model_tmp.add_module(str(len(model_tmp)), deeper)
            model_tmp.add_module(
                str(len(model_tmp)),
                nn.BatchNorm2d(
                    num_channels_skip[i]
                    + (
                        num_channels_up[i + 1]
                        if i < last_scale
                        else num_channels_down[i]
                    )
                ),
            )
            if num_channels_skip[i] != 0:
                skip.add_module(
                    str(len(skip)),
                    ConvBNAct(
                        input_depth,
                        num_channels_skip[i],
                        filter_skip_size,
                        act_fun=act_fun,
                        is_gated=hparams.use_gated_conv,
                    ),
                )

            deeper.add_module(
                str(len(deeper)),
                ConvBNAct(
                    input_depth,
                    num_channels_down[i],
                    filter_size_down,
                    stride=2,
                    act_fun=act_fun,
                    is_gated=hparams.use_gated_conv,
                ),
            )

            deeper.add_module(
                str(len(deeper)),
                ConvBNAct(
                    num_channels_down[i],
                    num_channels_down[i],
                    filter_size_down,
                    act_fun=act_fun,
                    is_gated=hparams.use_gated_conv,
                ),
            )
            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add_module(str(len(deeper)), deeper_main)
                k = num_channels_up[i + 1]
            deeper.add_module(
                str(len(deeper)), nn.Upsample(scale_factor=2, mode=upsample_mode)
            )
            model_tmp.add_module(
                str(len(model_tmp)),
                ConvBNAct(
                    num_channels_skip[i] + k,
                    num_channels_up[i],
                    filter_size_up,
                    act_fun=act_fun,
                ),
            )

            if need1x1_up:
                model_tmp.add_module(
                    str(len(model_tmp)),
                    ConvBNAct(
                        num_channels_up[i], num_channels_up[i], 1, act_fun=act_fun
                    ),
                )
            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        model.add_module(
            str(len(model)), nn.Conv2d(num_channels_up[0], num_output_channels, 1)
        )
        if need_sigmoid:
            model.add_module(str(len(model)), nn.Sigmoid())
        self.model = model

    def forward(self, x):
        return self.model(x)


class DeepImagePrior(pl.LightningModule):
    def __init__(self, hparams):
        super(DeepImagePrior, self).__init__()
        self.hparams = hparams
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.loss_type = hparams.loss_type
        # self.data_root = "data/sub_data"
        self.data_root = "data"
        # self.data_root = Path(
        # "C:\\Users\\zhang\\Desktop\\Code\\inpaint\\pytorch-lightning-seed\\data\\sub_data"
        # )
        self.saved_output = None
        self.predict_output = None
        self.show_masked_once = False

        self.model = DIP(hparams)

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
        return DataLoader(DeepImagePriorDataset(self.data_root), batch_size=1)

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
        parser.add_argument("--learning_rate", default=0.01, type=float)
        parser.add_argument("--batch_size", default=1, type=int)
        parser.add_argument("--depth", default=6, type=int)
        # parser.add_argument("--loss_type", default="psnr", type=str)
        # parser.add_argument("--loss_type", default="mse", type=str)
        # parser.add_argument("--loss_type", default="weighted_psnr", type=str)
        parser.add_argument("--loss_type", default="weighted_mse", type=str)
        parser.add_argument("--ckpt_path", default="dip_model", type=str)
        parser.add_argument("--use_gated_conv", action="store_true")

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=5000, type=int)

        return parser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = DeepImagePrior.add_model_specific_args(parser)

    hparams = parser.parse_args()

    DeepImagePrior(hparams)
