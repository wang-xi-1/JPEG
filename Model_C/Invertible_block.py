import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class HIN(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HIN, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out_1 = self.norm(out_1)
            out = torch.cat([out_1, out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        )

    def forward(self, x):

        return self.main(x) + x


#----------------------------------------------------------------------------------------------
def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'RES':
            return ResBlock(channel_in, channel_out)
        if net_structure == 'HIN':
            return HIN(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, subnet_constructor=subnet('RES'), clamp=0.8):   ################  split_channel一般设为channel_num的一半
        super(InvBlock, self).__init__()

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x1, x2):

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)
        return out

'''
if __name__ == "__main__":
    model = InvBlock(channel_num=128, channel_split_num=64)
    input = torch.randn(1, 64, 256, 256)
    output = model(input, input)
    print('-'*50)
    print(output.shape)
'''