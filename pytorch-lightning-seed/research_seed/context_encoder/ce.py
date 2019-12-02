"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
from typing import List, Any
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from research_seed.common import ConvBNAct


class Generator(nn.Module):
    def __init__(self, in_channel):
        super(Generator, self).__init__()

        self.encoder = Encoder(in_channel)
        self.mid = nn.Sequential(
            nn.Conv2d(512, 4000, 4, stride=2, bias=True),
            nn.BatchNorm2d(4000),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = Decoder(4000)
        self.model = nn.Sequential(
            self.encoder,
            self.mid,
            self.decoder
        )

    def forward(self, x):
        # x = self.encoder(x)
        # x = self.mid(x)
        # x = self.decoder(x)
        # return x
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        out_channels = [64, 64, 128, 256, 512]
        blocks = OrderedDict()
        for i in range(len(out_channels)):
            if i == 0:
                blocks["encoder_{}".format(i)] = ConvBNAct(
                    in_channel, out_channels[i], 4, stride=2
                )
            else:
                blocks["encoder_{}".format(i)] = ConvBNAct(
                    out_channels[i - 1], out_channels[i], 4, stride=2
                )
        # self.blocks = blocks
        self.model = nn.Sequential(blocks)
        pass

    def forward(self, x):
        # for key, value in self.blocks.items():
        #     x = value(x)
        #     print(x.shape)
        # return x
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_channel):
        super(Decoder, self).__init__()
        out_channels = [3, 64, 128, 256, 512]
        blocks = OrderedDict()
        for i in reversed(range(len(out_channels))):
            if i == len(out_channels) - 1:
                blocks["decoder_{}".format(i)] = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_channels[i], 4),
                    nn.BatchNorm2d(out_channels[i]),
                    nn.ReLU(inplace=True)
                )
            else:
                blocks["decoder_{}".format(i)] = ConvBNAct(
                    out_channels[i + 1], out_channels[i], 4, stride=2, act_fun="ReLU", is_deconv=True
                )
        # self.blocks = blocks
        self.model = nn.Sequential(blocks)
        pass

    def forward(self, x):
        # for key, value in self.blocks.items():
        #     x = value(x)
        #     print(x.shape)
        # return x
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()
        out_channels = [64, 128, 256, 512]
        blocks = OrderedDict()
        for i in range(len(out_channels)):
            if i == 0:
                blocks["discriminator_{}".format(i)] = ConvBNAct(
                    in_channel, out_channels[i], 4, stride=2
                )
            else:
                blocks["discriminator_{}".format(i)] = ConvBNAct(
                    out_channels[i - 1], out_channels[i], 4, stride=2
                )
        blocks["discriminator_fc"] = nn.Sequential(
            nn.Conv2d(out_channels[-1], 1, 4),
            nn.Sigmoid()
        )
        # self.blocks = blocks
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        # for key, value in self.blocks.items():
        #     x = value(x)
        #     print(x.shape)
        # return x
        return self.model(x)


class ContextEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(ContextEncoder, self).__init__()
        self.hparams = hparams

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(3)
        self.discriminator = Discriminator(3)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_i == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self.forward(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs), valid
            )
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_i == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr * 0.1, betas=(b1, b2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        z = torch.randn(8, self.hparams.latent_dim)
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f"generated_images", grid, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parent_parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parent_parser.add_argument("--b1", type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        parent_parser.add_argument("--latent_dim", type=int, default=1,
                        help="dimensionality of the latent space")
        parent_parser.add_argument("--b2", type=float, default=0.999,
                            help="adam: decay of first order momentum of gradient")
        return parent_parser


if __name__ == "__main__":
    gen = Generator(3)(torch.rand(2, 3, 128, 128))
    label = Discriminator(3)(gen)
