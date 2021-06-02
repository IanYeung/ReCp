import functools

import torch
from torch import nn as nn

from basicsr.archs.arch_util import SFTLayer, ResidualBlockNoBN, ResBlock_with_SFT, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

from compressai.layers import (
    AttentionBlock,
    ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride,
    conv3x3, subpel_conv3x3,
)
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420
)
from compressai.ops import ste_round


class SimpleBlock(nn.Module):

    def __init__(self, depth=3, n_channels=64, in_nc=3, out_nc=64, kernel_size=3, padding=1, bias=True):
        super(SimpleBlock, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_nc, n_channels, kernel_size=(kernel_size, kernel_size), padding=(padding, padding), bias=bias)
        )
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size=(kernel_size, kernel_size),
                          padding=(padding, padding), bias=bias)
            )
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(
            nn.Conv2d(n_channels, out_nc, kernel_size=(kernel_size, kernel_size),
                      padding=(padding, padding), bias=bias)
        )
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@ARCH_REGISTRY.register()
class EncoderDecoder(nn.Module):

    def __init__(self, depth=2, nf=64, num_in_ch=3, num_out_ch=3, color_space='rgb'):
        super(EncoderDecoder, self).__init__()

        # encoder
        self.conv_block_s1 = SimpleBlock(depth=depth, n_channels=1 * nf, in_nc=1 * num_in_ch, out_nc=nf, kernel_size=3)
        self.pool1 = nn.Conv2d(nf, 2 * nf, (3, 3), (2, 2), (1, 1), bias=True)

        self.conv_block_s2 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=2 * nf, out_nc=2 * nf, kernel_size=3)
        self.pool2 = nn.Conv2d(2 * nf, 4 * nf, (3, 3), (2, 2), (1, 1), bias=True)

        # intermediate processing
        self.conv_block_s3 = SimpleBlock(depth=depth, n_channels=4 * nf, in_nc=4 * nf, out_nc=4 * nf, kernel_size=3)

        # decoder
        self.up1 = nn.ConvTranspose2d(4 * nf, 2 * nf, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)
        self.conv_block_s4 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=2 * nf, out_nc=2 * nf, kernel_size=3)

        self.up2 = nn.ConvTranspose2d(2 * nf, 1 * nf, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)
        self.conv_block_s5 = SimpleBlock(depth=depth, n_channels=1 * nf, in_nc=1 * nf, out_nc=num_out_ch, kernel_size=3)

        self.color_space = color_space

    def forward(self, x):

        if self.color_space == 'ycbcr':
            x = rgb2ycbcr(x)

        # encoder
        x_s1 = self.conv_block_s1(x)     # 064, H/1, W/1
        x_s2 = self.pool1(x_s1)          # 128, H/2, W/2
        x_s2 = self.conv_block_s2(x_s2)  # 128, H/2, W/2
        x_s3 = self.pool2(x_s2)          # 256, H/4, W/4

        x_s3 = self.conv_block_s3(x_s3)  # 256, H/4, W/4

        # decoder
        out = self.up1(x_s3)             # 128, H/2, W/2
        out = self.conv_block_s4(out)    # 128, H/2, W/2
        out = self.up2(out)              # 064, H/1, W/1
        out = self.conv_block_s5(out)    # out, H/1, W/1
        out += x

        if self.color_space == 'ycbcr':
            out = ycbcr2rgb(out)

        return out


@ARCH_REGISTRY.register()
class BIC(nn.Module):

    def __init__(self, nf=64, num_in_ch=3, num_out_ch=3, color_space='rgb'):
        super(BIC, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(num_in_ch, nf, stride=2),
            ResidualBlock(nf, nf),
            ResidualBlockWithStride(nf, nf, stride=2),
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockWithStride(nf, nf, stride=2),
            ResidualBlock(nf, nf),
            conv3x3(nf, nf, stride=2),
            AttentionBlock(nf),
        )

        self.decoder = nn.Sequential(
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            ResidualBlock(nf, nf),
            subpel_conv3x3(nf, num_out_ch, 2),
        )

        self.color_space = color_space

    def forward(self, inp):
        if self.color_space == 'ycbcr':
            inp = rgb2ycbcr(inp)
        hid = self.encoder(inp)
        out = self.decoder(hid)
        if self.color_space == 'ycbcr':
            out = rgb2ycbcr(out)
        return out


@ARCH_REGISTRY.register()
class BICQ(nn.Module):

    def __init__(self, nf=64, num_in_ch=3, num_out_ch=3, color_space='rgb'):
        super(BICQ, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(num_in_ch, nf, stride=2),
            ResidualBlock(nf, nf),
            ResidualBlockWithStride(nf, nf, stride=2),
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockWithStride(nf, nf, stride=2),
            ResidualBlock(nf, nf),
            conv3x3(nf, nf, stride=2),
            AttentionBlock(nf),
        )

        self.decoder = nn.Sequential(
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            AttentionBlock(nf),
            ResidualBlock(nf, nf),
            ResidualBlockUpsample(nf, nf, 2),
            ResidualBlock(nf, nf),
            subpel_conv3x3(nf, num_out_ch, 2),
        )

        self.color_space = color_space

    def forward(self, inp):
        if self.color_space == 'ycbcr':
            inp = rgb2ycbcr(inp)
        hid = self.encoder(inp)
        hid = ste_round(hid)
        out = self.decoder(hid)
        if self.color_space == 'ycbcr':
            out = rgb2ycbcr(out)
        return out


@ARCH_REGISTRY.register()
class CondUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(CondUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.SFT_layer1 = SFTLayer()
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)

        basic_block = functools.partial(ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = make_layer(basic_block, 2)
        self.recon_trunk2 = make_layer(basic_block, 8)
        self.recon_trunk3 = make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = SFTLayer()
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc = 3
        cond_nf = 64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(nf, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(nf, nf, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(nf, out_nc, 1),
                                      )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # x[0]: img; x[1]: cond
        mask = self.mask_est(x[0])

        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2

        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))

        out = self.act(self.up_conv2(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask * x[0] + out
        return out