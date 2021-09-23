import functools
import math
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import Prediction, Rescaler
from basicsr.utils.registry import ARCH_REGISTRY

from torchjpeg.dct import blockify, deblockify

from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel

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
            out = ycbcr2rgb(out)
        return out


@ARCH_REGISTRY.register()
class SingleFrameCompressor(CompressionModel):

    def __init__(self, num_ch=1, search_size=21, block_size=4, color='RGB'):
        super().__init__(entropy_bottleneck_channels=1)

        self.C_f4 = torch.tensor([[1,  1,  1,  1],
                                  [2,  1, -1, -2],
                                  [1, -1, -1,  1],
                                  [1, -2,  2, -1]],
                                 dtype=torch.float32)

        self.C_i4 = torch.tensor([[  1,   1,    1,    1],
                                  [  1, 1/2, -1/2,   -1],
                                  [  1,  -1,   -1,    1],
                                  [1/2,  -1,    1, -1/2]],
                                 dtype=torch.float32)

        self.S_f4 = torch.tensor([[1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                                  [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10],
                                  [1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                                  [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10]],
                                 dtype=torch.float32)

        self.S_i4 = torch.tensor([[1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                                  [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5],
                                  [1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                                  [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5]],
                                 dtype=torch.float32)

        self.Q_step_table = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125]
        self.search_size = search_size
        self.block_size = block_size
        self.color = color

        self.rescaler = Rescaler()
        self.min = None
        self.max = None

        # Entropy Bottleneck
        for i in range(self.block_size * self.block_size):
            setattr(self, 'entropy_bottleneck_{:02d}'.format(i), EntropyBottleneck(num_ch))

    def forward(self, curr_frame, qp, training=False, beta=100, debug=False):
        if self.color == 'RGB':
            pred_frame, _ = Prediction.intra_prediction_ste_rgb(ori_images, ori_images,
                                                                search_size=self.search_size,
                                                                block_size=self.block_size,
                                                                beta=beta)
        else:
            pred_frame, _ = Prediction.intra_prediction_ste_y(ori_images, ori_images,
                                                              search_size=self.search_size,
                                                              block_size=self.block_size,
                                                              beta=beta)
        residual = curr_frame - pred_frame

        # rescale residual image to [0, 255]
        # residual = self.rescaler.fwd_rescale_pt(residual) * 255.0
        self.min, self.max = torch.min(residual), torch.max(residual)
        residual = (residual - self.min) / (self.max - self.min) * 255.0
        if debug:
            ori_res = residual

        residual, likehihood_list = self.forward_residual(residual, qp, training)

        # rescale residual image back
        if debug:
            com_res = residual
        # residual = self.rescaler.bwd_rescale_pt(residual / 255.0)
        residual = (residual / 255.0 * (self.max - self.min) + self.min)

        curr_frame = pred_frame + residual
        if debug:
            return curr_frame, likehihood_list, ori_res, com_res
        else:
            return curr_frame, likehihood_list

    def forward_residual(self, ori_images, qp, training=False):

        self.C_f4 = self.C_f4.to(ori_images.device)
        self.C_i4 = self.C_i4.to(ori_images.device)
        self.S_f4 = self.S_f4.to(ori_images.device)
        self.S_i4 = self.S_i4.to(ori_images.device)

        B, C, H, W = ori_images.shape

        X = blockify(ori_images, size=self.block_size)
        C_f4, S_f4 = self.C_f4.expand(*X.shape), self.S_f4.expand(*X.shape)

        # forward transform
        Y = C_f4 @ X @ C_f4.transpose(-1, -2) * S_f4

        # quantization
        Q_step = self.Q_step_table[qp % 6] * (2 ** math.floor(qp / 6))
        Y_h = Y / Q_step

        inp_subbands_list = self._extract_subband(Y_h, H, W)

        # dct_subbands_list = list(map(torch.round, dct_subbands_list))
        out_subband_list, likehihood_list = [], []
        for i in range(self.block_size * self.block_size):
            subband, likelihood = \
                getattr(self, 'entropy_bottleneck_{:02d}'.format(i))(inp_subbands_list[i], training=training)
            out_subband_list.append(subband)
            likehihood_list.append(likelihood)

        Y_h = self._combine_subband(out_subband_list)

        Y_h = Y_h * Q_step

        # backward transform
        C_i4, S_i4 = self.C_i4, self.S_i4
        Z = torch.round(C_i4.transpose(-1, -2) @ (Y_h * S_i4) @ C_i4)

        com_images = deblockify(Z, size=(H, W))  # [B, C, H, W]
        return com_images, likehihood_list

    def _extract_subband(self, Y_h, H, W):
        dct_subband_list = []
        Y_h = deblockify(Y_h, size=(H, W))  # [B, C, H, W]
        Y_sub = F.pixel_unshuffle(Y_h, downscale_factor=4)  # [B, C, H//4, W//4]
        for i in range(self.block_size * self.block_size):
            dct_subband_list.append(Y_sub[:, (i)*Y_h.size(1):(i+1)*Y_h.size(1), :, :])
        return dct_subband_list

    def _combine_subband(self, dct_subband_list):
        dct_subband = torch.cat(dct_subband_list, dim=1)  # [B, C, H//4, W//4]
        Y_h = F.pixel_shuffle(dct_subband, upscale_factor=4)  # [B, C, H, W]
        Y_h = blockify(Y_h, size=self.block_size)
        return Y_h

    def _foward_divisive_normalization(self, dct_subband_list):
        return dct_subband_list

    def _inverse_divisive_normalization(self, dct_subband_list):
        return dct_subband_list


@ARCH_REGISTRY.register()
class DoubleFrameCompressor(CompressionModel):

    def __init__(self, num_ch=1, search_size=21, block_size=4, color='RGB'):
        super().__init__(entropy_bottleneck_channels=1)

        self.C_f4 = torch.tensor([[1,  1,  1,  1],
                                  [2,  1, -1, -2],
                                  [1, -1, -1,  1],
                                  [1, -2,  2, -1]],
                                 dtype=torch.float32)

        self.C_i4 = torch.tensor([[  1,   1,    1,    1],
                                  [  1, 1/2, -1/2,   -1],
                                  [  1,  -1,   -1,    1],
                                  [1/2,  -1,    1, -1/2]],
                                 dtype=torch.float32)

        self.S_f4 = torch.tensor([[1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                                  [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10],
                                  [1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                                  [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10]],
                                 dtype=torch.float32)

        self.S_i4 = torch.tensor([[1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                                  [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5],
                                  [1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                                  [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5]],
                                 dtype=torch.float32)

        # self.Q_step_table = [0.6250, 0.6875, 0.8125, 0.8750, 1.0000, 1.1250]
        # self.Q_step_table = [0.6250, 0.7031, 0.7812, 0.8984, 0.9766, 1.1328]
        self.Q_step_table = [0.6423, 0.6917, 0.7906, 0.8894, 0.9882, 1.1364]

        self.search_size = search_size
        self.block_size = block_size
        self.color = color

        self.rescaler = Rescaler()
        self.min = None
        self.max = None

        # Entropy Bottleneck
        for i in range(self.block_size * self.block_size):
            setattr(self, 'entropy_bottleneck_{:02d}'.format(i), EntropyBottleneck(num_ch))

    def forward(self, curr_frame, prev_frame, qp, training=False, mode='inter', beta=100, debug=False):
        if mode == 'inter':
            if self.color == 'RGB':
                pred_frame, flow = Prediction.inter_prediction_ste_rgb(curr_frame, prev_frame,
                                                                      search_size=self.search_size,
                                                                      block_size=self.block_size,
                                                                      beta=beta)
            else:
                pred_frame, flow = Prediction.inter_prediction_ste_y(curr_frame, prev_frame,
                                                                    search_size=self.search_size,
                                                                    block_size=self.block_size,
                                                                    beta=beta)
        else:
            if self.color == 'RGB':
                pred_frame, flow = Prediction.intra_prediction_ste_rgb(curr_frame, curr_frame,
                                                                      search_size=self.search_size,
                                                                      block_size=self.block_size,
                                                                      beta=beta)
            else:
                pred_frame, flow = Prediction.intra_prediction_ste_y(curr_frame, curr_frame,
                                                                    search_size=self.search_size,
                                                                    block_size=self.block_size,
                                                                    beta=beta)
        residual = curr_frame - pred_frame

        # rescale residual image to [0, 255]
        # residual = self.rescaler.fwd_rescale_pt(residual) * 255.0
        self.min, self.max = torch.min(residual), torch.max(residual)
        residual = (residual - self.min) / (self.max - self.min) * 255.0
        if debug:
            ori_res = residual

        residual, likehihood_list = self.forward_residual(residual, qp, training)

        # rescale residual image back
        if debug:
            com_res = residual
        # residual = self.rescaler.bwd_rescale_pt(residual / 255.0)
        residual = (residual / 255.0 * (self.max - self.min) + self.min)

        curr_frame = pred_frame + residual
        if debug:
            return curr_frame, flow, likehihood_list, ori_res, com_res
        else:
            return curr_frame, flow, likehihood_list

    def forward_residual(self, ori_images, qp, training=False):

        self.C_f4 = self.C_f4.to(ori_images.device)
        self.C_i4 = self.C_i4.to(ori_images.device)
        self.S_f4 = self.S_f4.to(ori_images.device)
        self.S_i4 = self.S_i4.to(ori_images.device)

        B, C, H, W = ori_images.shape

        X = blockify(ori_images, size=self.block_size)
        C_f4, S_f4 = self.C_f4.expand(*X.shape), self.S_f4.expand(*X.shape)

        # forward transform
        Y = C_f4 @ X @ C_f4.transpose(-1, -2) * S_f4

        # quantization
        Q_step = self.Q_step_table[qp % 6] * (2 ** math.floor(qp / 6))
        Y_h = Y / Q_step

        inp_subband_list = self._extract_subband(Y_h, H, W)

        out_subband_list , likehihood_list = list(map(torch.round, inp_subband_list)), None
        # out_subband_list, likehihood_list = [], []
        # for i in range(self.block_size * self.block_size):
        #     subband, likelihood = \
        #         getattr(self, 'entropy_bottleneck_{:02d}'.format(i))(inp_subband_list[i], training=training)
        #     out_subband_list.append(subband)
        #     likehihood_list.append(likelihood)

        Y_h = self._combine_subband(out_subband_list)

        Y_h = Y_h * Q_step

        # backward transform
        C_i4, S_i4 = self.C_i4, self.S_i4
        Z = C_i4.transpose(-1, -2) @ (Y_h * S_i4) @ C_i4

        com_images = deblockify(Z, size=(H, W))  # [B, C, H, W]
        return com_images, likehihood_list

    def _extract_subband(self, Y_h, H, W):
        dct_subband_list = []
        Y_h = deblockify(Y_h, size=(H, W))  # [B, C, H, W]
        Y_sub = F.pixel_unshuffle(Y_h, downscale_factor=4)  # [B, C, H//4, W//4]
        for i in range(self.block_size * self.block_size):
            dct_subband_list.append(Y_sub[:, (i)*Y_h.size(1):(i+1)*Y_h.size(1), :, :])
        return dct_subband_list

    def _combine_subband(self, dct_subband_list):
        dct_subband = torch.cat(dct_subband_list, dim=1)  # [B, C, H//4, W//4]
        Y_h = F.pixel_shuffle(dct_subband, upscale_factor=4)  # [B, C, H, W]
        Y_h = blockify(Y_h, size=self.block_size)
        return Y_h

    def _foward_divisive_normalization(self, dct_subband_list):
        return dct_subband_list

    def _inverse_divisive_normalization(self, dct_subband_list):
        return dct_subband_list