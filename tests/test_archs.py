import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from basicsr.archs import build_network
from basicsr.utils.matlab_functions import bgr2ycbcr, ycbcr2bgr


def get_encoderdecoder():
    opt = {'type': 'EncoderDecoder', 'depth': 2, 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3, 'color_space': 'rgb'}
    net = build_network(opt)
    return net


def get_bic():
    opt = {'type': 'BIC', 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3, 'color_space': 'rgb'}
    net = build_network(opt)
    return net


def get_basicvsr():
    opt = {'type': 'BasicVSR', 'num_feat': 64, 'num_block': 30, 'scale': 4, 'spynet_path': None}
    net = build_network(opt)
    return net


def get_unet2d():
    opt = {'type': 'UNet2D', 'in_channels': 3, 'out_channels': 3, 'num_levels': 2}
    net = build_network(opt)
    return net


def get_unet3d():
    opt = {'type': 'UNet3D', 'in_channels': 3, 'out_channels': 3, 'num_levels': 2}
    net = build_network(opt)
    return net


def get_resunet3d():
    opt = {'type': 'ResidualUNet3D', 'in_channels': 3, 'out_channels': 3, 'num_levels': 2}
    net = build_network(opt)
    return net


def get_fstrn():
    opt = {'type': 'FSTRN', 'ks': 3, 'nf': 64}
    net = build_network(opt)
    return net


def get_condunet():
    opt = {'type': 'CondUNet', 'in_nc': 3, 'out_nc': 3, 'nf': 64}
    net = build_network(opt)
    return net


def get_compressor():
    opt = {'type': 'FrameCompressor', 'num_ch': 1, 'block_size': 4}
    net = build_network(opt)
    return net


if __name__ == '__main__':

    device = torch.device('cpu')

    # inp = torch.randn(1, 3, 128, 224).to(device)
    # # inp = torch.randn(1, 7, 3, 128, 224).to(device)
    # net = get_condunet().to(device)
    # out = net((inp, inp))
    # print(out.shape)

    net = get_compressor().to(device)

    block_size = 4
    img = cv2.imread('/home/xiyang/Downloads/0100/im1.png')
    ori_img = bgr2ycbcr(img, y_only=True)  # [H, W]
    plt.imshow(ori_img, cmap='gray')
    plt.title('Original Image')
    plt.show()

    ori_images = torch.from_numpy(ori_img).unsqueeze(dim=0).unsqueeze(dim=0).float()  # [B, 1, H, W]

    com_images, likelihoods = net(ori_images, qp=10)  # [B, 1, H, W]

    com_img = (com_images.detach().squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8).squeeze()  # [H, W]
    plt.imshow(com_img, cmap='gray')
    plt.title('Compressed Image')
    plt.show()

    print(np.mean(np.abs(com_img - ori_img)))
    plt.imshow(np.abs(com_img - ori_img), cmap='gray')
    plt.title('Difference Image')
    plt.show()

    # net = get_compressor().to(device)
    #
    # block_size = 4
    # ori_img = cv2.imread('/home/xiyang/Downloads/0100/im1.png')[:, :, [2, 1, 0]]
    # plt.imshow(ori_img)
    # plt.title('Original Image')
    # plt.show()
    #
    # ori_images = torch.from_numpy(ori_img).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 3, H, W]
    #
    # com_images, likelihoods = net(ori_images, qp=30)  # [B, 3, H, W]
    #
    # com_img = (com_images.detach().squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8)  # [H, W, 3]
    # plt.imshow(com_img)
    # plt.title('Compressed Image')
    # plt.show()
