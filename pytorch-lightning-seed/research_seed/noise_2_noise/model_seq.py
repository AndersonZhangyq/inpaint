import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Common defination to describe the quaility of image, the bigger is better.
def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = torch.clamp(y_pred, 0.0, 255.0)  # clip the value to 0.0-255.0
    return 10.0 * torch.log10(
        (max_pixel ** 2) / (torch.mean(torch.pow((y_pred - y_true), 2)))
    )  # PSNR formula


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.PReLU(),
            nn.Conv2d(output_dim, output_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.block(x) + x


class SRRnetModule(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64, resunit_num=16):
        # I don't know the size of input, how to calculate the padding size?
        super(SRRnetModule, self).__init__()
        self.input_channel_num = input_channel_num
        self.feature_dim = feature_dim
        self.resunit_num = resunit_num

        self._init_weights()
        self.head = nn.Sequential(
            nn.Conv2d(self.input_channel_num, self.feature_dim, 3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.model = self.build_model()
        self.rest = nn.Conv2d(
            self.feature_dim, self.input_channel_num, 3, stride=1, padding=1
        )

    def _init_weights(self):
        """ Using kaiming initialization, he_normal in keras"""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        identity = self.head(x)
        x = self.model(identity)
        x += identity
        x = self.rest(x)
        return x

    def build_model(self):

        blocks = []

        for i in range(self.resunit_num):
            blocks.append(ResidualBlock(self.feature_dim, self.feature_dim))

        blocks.append(
            nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.feature_dim),
            )
        )
        return nn.Sequential(*blocks)


class Convblock(nn.Module):
    def __init__(self, input_channel, feature_dim):
        super(Convblock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, feature_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self._init_weights()

    def forward(self, inputs):
        return self.block(inputs)

    def _init_weights(self):
        """ Using kaiming initialization, he_normal in keras"""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, input_channel_num=3, out_channels=3):

        super(UNet, self).__init__()
        self.block1 = Convblock(3, 64)

        # self.blockt = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=(2,2)),
        #     Convblock(64,64)
        # )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)), Convblock(64, 128)
        )

        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)), Convblock(128, 256)
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)), Convblock(256, 512)
        )

        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)), Convblock(512, 1024)
        )

        self.block6 = nn.Sequential(nn.Upsample(scale_factor=(2)))

        self.block7 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, stride=2, padding=(5, 5)),
            nn.Upsample(scale_factor=(2)),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, stride=2, padding=(9, 9)),
            nn.Upsample(scale_factor=(2)),
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, stride=2, padding=(17, 17)),
            nn.Upsample(scale_factor=(2)),
        )

        self.block10 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, stride=2, padding=(33, 33)),
            nn.Conv2d(64, 3, 3, stride=2, padding=(33, 33)),
        )

        self._init_weights()

    def _init_weights(self):
        """ Using kaiming initialization, he_normal in keras"""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, inputs):
        n1 = self.block1(inputs)
        n2 = self.block2(n1)
        n3 = self.block3(n2)
        n4 = self.block4(n3)
        x = self.block5(n4)

        m4 = self.block6(x)
        c4 = torch.cat((n4, m4), dim=1)
        m3 = self.block7(c4)
        c3 = torch.cat((n3, m3), dim=1)
        m2 = self.block8(c3)
        c2 = torch.cat((n2, m2), dim=1)
        m1 = self.block9(c2)
        c1 = torch.cat((n1, m1), dim=1)
        # result = torch.clamp(self.block10(c1),min=0,max=255)
        result = self.block10(c1)

        return result


class L0Loss(nn.Module):
    def __init__(self):
        """Initialization"""
        super(L0Loss, self).__init__()
        self.gamma = torch.Tensor([2]).cuda()

    def forward(self, y_true, y_pred):
        true = y_true.cuda()
        pred = y_pred.cuda()
        loss = torch.pow(torch.abs(true - pred) + 1e-8, self.gamma).cuda()
        return loss.mean()
        # loss = ((pred - true) ** 2) / ((pred + 0.01) ** 2)
        # return torch.mean(loss.view(-1))
