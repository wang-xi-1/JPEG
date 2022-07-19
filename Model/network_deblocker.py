from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from Model.network_E import encoder
from Model.Invertible_block import InvBlock
from Model import common
Encoder_path = ''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.1):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU, res_scale=1):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.1, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class DeblockNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 64, 64, 64], nb=2):
        super(DeblockNet, self).__init__()

        self.E = encoder()
        # self.E.load_state_dict(torch.load(Encoder_path))
        for name, param in self.E.named_parameters():
            param.requires_grad = False
        print("Total number of paramerters in encorder networks is {} ".format(sum(x.numel() for x in self.E.parameters())))
        print("Total number of requires_grad paramerters in encorder networks is {} "
              .format(sum(p.numel() for p in self.E.parameters() if p.requires_grad)))

        self.conv2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv3 = nn.Conv2d(256, 64, 1, 1, 0)
        self.conv = conv(in_nc, nc[0], bias=True, mode='C')

        self.mid1 = sequential(
            *[ResBlock(nc[3]*2, nc[3]*2) for _ in range(nb)])

        self.mid2 = sequential(
            *[ResBlock(nc[3]*2, nc[3]*2) for _ in range(nb)],
              nn.Conv2d(nc[3]*2, nc[3], 1, 1, 0))

        self.invertible1 = InvBlock(channel_num=128, channel_split_num=64)
        self.invertible2 = InvBlock(channel_num=128, channel_split_num=64)
        self.invertible3 = InvBlock(channel_num=128, channel_split_num=64)

        self.m3 = sequential(nn.Conv2d(nc[2] * 2, nc[3], kernel_size=3, stride=1, padding=1),
                                *[_ResGroup(conv=default_conv, n_feats=nc[2], kernel_size=3) for _ in range(nb)])

        self.m2 = sequential(nn.Conv2d(nc[2] * 2, nc[3], kernel_size=3, stride=1, padding=1),
                                *[_ResGroup(conv=default_conv, n_feats=nc[1], kernel_size=3) for _ in range(nb * 2)])

        self.m1 = sequential(nn.Conv2d(nc[2] * 2, nc[3], kernel_size=3, stride=1, padding=1),
                                *[_ResGroup(conv=default_conv, n_feats=nc[0], kernel_size=3) for _ in range(nb * 2)])

        self.paramid = PyramidPooling(nc[0], nc[0])
        self.tail = conv(nc[0], out_nc, bias=True, mode='C')

        self.AFFs = nn.ModuleList([
            AFF(nc[0]*3, nc[0]),
            AFF(nc[0]*3, nc[2])
        ])

    def forward(self, x):

        input = x
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x0, x1, x2, x3, _ = self.E(x)
        x0 = F.interpolate(x0, size=x.shape[2:])
        x1 = F.interpolate(x1, size=x.shape[2:])
        x2 = F.interpolate(x2, size=x.shape[2:])
        x2 = self.conv2(x2)
        x3 = F.interpolate(x3, size=x.shape[2:])
        x3 = self.conv3(x3)

        x01 = F.interpolate(x0, scale_factor=1)
        x12 = F.interpolate(x1, scale_factor=1)
        x21 = F.interpolate(x2, scale_factor=1)
        x32 = F.interpolate(x3, scale_factor=1)

        x1 = self.AFFs[0](x1, x01, x21)
        x2 = self.AFFs[1](x2, x32, x12)

        x = self.conv(x)
        tmp = torch.cat([x, x3], dim=1)

        x = self.mid1(tmp)
        x = self.mid2(x)  # b, 64, h, w

        tmp3 = self.invertible1(x, x3)
        x = self.m3(tmp3)

        tmp2 = self.invertible2(x, x2)
        x = self.m2(tmp2)

        tmp1 = self.invertible3(x, x1)
        x = self.m1(tmp1)

        x = self.paramid(x)
        x = self.tail(x)

        out = x[..., :h, :w]

        return out + input


if __name__ == "__main__":
    x = torch.randn(2, 1, 127, 128)        #.cuda()#.to(torch.device('cuda'))
    fbar=DeblockNet()
    print(fbar)
    y = fbar(x)
    print(y.shape)
    print('-' * 50)
    print('#generator parameters:', sum(param.numel() for param in fbar.parameters()))
