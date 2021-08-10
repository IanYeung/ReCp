import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_with_SFT(nn.Module):
    def __init__(self, nf=64):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        default_init_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = F.relu(self.conv1(fea), inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return x[0] + fea, x[1]


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class Prediction(nn.Module):
    """
    Differentiable intra/inter prediction module of H264 codec
    """
    def __init__(self, search_size=21, block_size=4):
        super(Prediction, self).__init__()
        self.search_size = search_size
        self.block_size = block_size

    def forward(self, im1, im2, mode='inter', soft=True):
        """
        Args:
            im1: [B, 1, H, W]  # neighbour frame
            im2: [B, 1, H, W]  # reference frame
            mode: 'inter' or 'intra'
        Returns:
            out: [B, 1, H, W]  # predicted frame
        """
        if mode == 'inter':
            return self.inter_prediction(im1, im2, self.search_size, self.block_size, soft)
        else:
            return self.intra_prediction(im1, im2, self.search_size, self.block_size, soft)

    @staticmethod
    def intra_prediction(im1, im2, search_size, block_size=4, soft=True):
        """
        Perform intra-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im1_pad = F.pad(im1, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                if i == search_size // 2 and j == search_size // 2:
                    continue
                search_list.append(im1_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im1_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        nbr = torch.cat(search_list, dim=1)
        if soft:
            idx = F.softmax(-100 * vol, dim=1)
        else:
            idx = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2)
        idx = idx.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)

        out = torch.sum(idx * nbr, dim=1, keepdim=True)
        return out

    @staticmethod
    def inter_prediction(im1, im2, search_size, block_size=4, soft=True):
        """
        Perform inter-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im2_pad = F.pad(im2, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                search_list.append(im2_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im2_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        nbr = torch.cat(search_list, dim=1)
        if soft:
            idx = F.softmax(-100 * vol, dim=1)
        else:
            idx = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2)
        idx = idx.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)

        out = torch.sum(idx * nbr, dim=1, keepdim=True)
        return out

    @staticmethod
    def intra_prediction_ste(im1, im2, search_size, block_size=4):
        """
        Perform intra-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im1_pad = F.pad(im1, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                if i == search_size // 2 and j == search_size // 2:
                    continue
                search_list.append(im1_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im1_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        nbr = torch.cat(search_list, dim=1)
        # implement argmin operation with straight through estimator
        idx_onehot = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2) + \
                     F.softmax(-100 * vol, dim=1).detach() - F.softmax(-100 * vol, dim=1)
        idx_onehot_expanded = idx_onehot.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
        # select top 1 patch
        out = torch.sum(idx_onehot_expanded * nbr, dim=1, keepdim=True)

        pos = torch.arange(0, vol.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(im1.device)
        pos_expanded = pos.expand(B, vol.shape[1], H // block_size, W // block_size)
        idx_min = torch.sum(pos_expanded * idx_onehot, dim=1, keepdim=False)  # [B, H, W]
        idx_u, idx_v = idx_min // search_size - pad_size, idx_min % search_size - pad_size
        flow = torch.stack([idx_u, idx_v], dim=-1)  # [B, H, W, 2]

        return out, flow

    @staticmethod
    def inter_prediction_ste(im1, im2, search_size, block_size=4):
        """
        Perform inter-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im2_pad = F.pad(im2, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                search_list.append(im2_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im2_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        nbr = torch.cat(search_list, dim=1)
        # implement argmin operation with straight through estimator
        idx_onehot = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2) + \
                     F.softmax(-100 * vol, dim=1).detach() - F.softmax(-100 * vol, dim=1)
        idx_onehot_expanded = idx_onehot.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
        # select top 1 patch
        out = torch.sum(idx_onehot_expanded * nbr, dim=1, keepdim=True)

        pos = torch.arange(0, vol.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(im1.device)
        pos_expanded = pos.expand(B, vol.shape[1], H // block_size, W // block_size)
        idx_min = torch.sum(pos_expanded * idx_onehot, dim=1, keepdim=False)  # [B, H, W]
        idx_u, idx_v = idx_min // search_size - pad_size, idx_min % search_size - pad_size
        flow = torch.stack([idx_u, idx_v], dim=-1)  # [B, H, W, 2]

        return out, flow

    @staticmethod
    def intra_prediction_rgb(im1, im2, search_size, block_size=4, soft=True):
        """
        Perform intra-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im1_pad = F.pad(im1, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                if i == search_size // 2 and j == search_size // 2:
                    continue
                search_list.append(im1_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im1_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        vol = vol[:, 0::3, :, :] + vol[:, 1::3, :, :] + vol[:, 2::3, :, :]
        nbr = torch.cat(search_list, dim=1)
        if soft:
            idx = F.softmax(-100 * vol, dim=1)
        else:
            idx = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2)
        idx = idx.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)

        out1 = torch.sum(idx * nbr[:, 0::3, :, :], dim=1, keepdim=True)
        out2 = torch.sum(idx * nbr[:, 1::3, :, :], dim=1, keepdim=True)
        out3 = torch.sum(idx * nbr[:, 2::3, :, :], dim=1, keepdim=True)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

    @staticmethod
    def inter_prediction_rgb(im1, im2, search_size, block_size=4, soft=True):
        """
        Perform inter-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im2_pad = F.pad(im2, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                search_list.append(im2_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im2_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        vol = vol[:, 0::3, :, :] + vol[:, 1::3, :, :] + vol[:, 2::3, :, :]
        nbr = torch.cat(search_list, dim=1)
        if soft:
            idx = F.softmax(-100 * vol, dim=1)
        else:
            idx = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2)
        idx = idx.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)

        out1 = torch.sum(idx * nbr[:, 0::3, :, :], dim=1, keepdim=True)
        out2 = torch.sum(idx * nbr[:, 1::3, :, :], dim=1, keepdim=True)
        out3 = torch.sum(idx * nbr[:, 2::3, :, :], dim=1, keepdim=True)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

    @staticmethod
    def intra_prediction_ste_rgb(im1, im2, search_size, block_size=4):
        """
        Perform intra-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im1_pad = F.pad(im1, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                if i == search_size // 2 and j == search_size // 2:
                    continue
                search_list.append(im1_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im1_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        vol = vol[:, 0::3, :, :] + vol[:, 1::3, :, :] + vol[:, 2::3, :, :]
        nbr = torch.cat(search_list, dim=1)
        # implement argmin operation with straight through estimator
        idx_onehot = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2) + \
                     F.softmax(-100 * vol, dim=1).detach() - F.softmax(-100 * vol, dim=1)
        idx_onehot_expanded = idx_onehot.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
        # select top 1 patch
        out1 = torch.sum(idx_onehot_expanded * nbr[:, 0::3, :, :], dim=1, keepdim=True)
        out2 = torch.sum(idx_onehot_expanded * nbr[:, 1::3, :, :], dim=1, keepdim=True)
        out3 = torch.sum(idx_onehot_expanded * nbr[:, 2::3, :, :], dim=1, keepdim=True)
        out = torch.cat([out1, out2, out3], dim=1)

        pos = torch.arange(0, vol.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(im1.device)
        pos_expanded = pos.expand(B, vol.shape[1], H // block_size, W // block_size)
        idx_min = torch.sum(pos_expanded * idx_onehot, dim=1, keepdim=False)  # [B, H, W]
        idx_u, idx_v = idx_min // search_size - pad_size, idx_min % search_size - pad_size
        flow = torch.stack([idx_u, idx_v], dim=-1)  # [B, H, W, 2]

        return out, flow

    @staticmethod
    def inter_prediction_ste_rgb(im1, im2, search_size, block_size=4):
        """
        Perform inter-frame prediction as in H264 codec
        Args:
            im1: Tensor of size [B, 1, H, W]
            im2: Tensor of size [B, 1, H, W]
            block_size: int
            search_size: int
            soft: True/False

        Returns:
            out: Tensor of size [B, 1, H, W]
        """
        B, C, H, W = im1.shape
        cost_volume = []
        search_list = []

        pad_size = search_size // 2
        im2_pad = F.pad(im2, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        for i in range(0, search_size):
            for j in range(0, search_size):
                search_list.append(im2_pad[:, :, i:i + H, j:j + W])
                cost_volume.append(F.avg_pool2d(torch.abs(im1 - im2_pad[:, :, i:i + H, j:j + W]), block_size))
        vol = torch.cat(cost_volume, dim=1)
        vol = vol[:, 0::3, :, :] + vol[:, 1::3, :, :] + vol[:, 2::3, :, :]
        nbr = torch.cat(search_list, dim=1)
        # implement argmin operation with straight through estimator
        idx_onehot = F.one_hot(torch.argmin(vol, dim=1), num_classes=vol.shape[1]).permute(0, 3, 1, 2) + \
                     F.softmax(-100 * vol, dim=1).detach() - F.softmax(-100 * vol, dim=1)
        idx_onehot_expanded = idx_onehot.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
        # select top 1 patch
        out1 = torch.sum(idx_onehot_expanded * nbr[:, 0::3, :, :], dim=1, keepdim=True)
        out2 = torch.sum(idx_onehot_expanded * nbr[:, 1::3, :, :], dim=1, keepdim=True)
        out3 = torch.sum(idx_onehot_expanded * nbr[:, 2::3, :, :], dim=1, keepdim=True)
        out = torch.cat([out1, out2, out3], dim=1)

        pos = torch.arange(0, vol.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(im1.device)
        pos_expanded = pos.expand(B, vol.shape[1], H // block_size, W // block_size)
        idx_min = torch.sum(pos_expanded * idx_onehot, dim=1, keepdim=False)  # [B, H, W]
        idx_u, idx_v = idx_min // search_size - pad_size, idx_min % search_size - pad_size
        flow = torch.stack([idx_u, idx_v], dim=-1)  # [B, H, W, 2]

        return out, flow


class Rescaler:
    """
    Tensor Rescaling
    """
    def __init__(self):
        self.min, self.max = None, None

    def fwd_rescale_pt(self, tensor):
        self.min, self.max = torch.min(tensor), torch.max(tensor)
        rescaled_tensor = (tensor - self.min) / (self.max - self.min)
        return rescaled_tensor

    def bwd_rescale_pt(self, tensor):
        rescaled_tensor = tensor * (self.max - self.min) + self.min
        return rescaled_tensor

    def fwd_rescale_np(self, tensor):
        self.min, self.max = np.min(tensor), np.max(tensor)
        rescaled_tensor = (tensor - self.min) / (self.max - self.min)
        return rescaled_tensor

    def bwd_rescale_np(self, tensor):
        rescaled_tensor = tensor * (self.max - self.min) + self.min
        return rescaled_tensor


if __name__ == '__main__':
    pass