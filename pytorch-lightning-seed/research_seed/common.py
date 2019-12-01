import torch
import torch.nn as nn
import numpy as np


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(x))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for input in inputs:
                diff2 = (input.size(2) - target_shape2) // 2
                diff3 = (input.size(3) - target_shape3) // 2
                inputs_.append(input[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def ConvBNAct(
    in_channel, out_channel, kernel_size, stride=1, act_fun="LeakyReLU", is_deconv=False
):
    to_pad = int((kernel_size - 1) / 2)
    conv = (
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding=to_pad)
        if is_deconv
        else nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=to_pad)
    )

    bn = nn.BatchNorm2d(out_channel)
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            act = nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "ReLU":
            act = nn.ReLU(inplace=True)
        elif act_fun == "Swish":
            act = Swish()
        elif act_fun == "ELU":
            act = nn.ELU(inplace=True)
        elif act_fun == "none":
            act = nn.Sequential()
        else:
            assert False
    else:
        assert False

    return nn.Sequential(conv, bn, act)
