import cv2
import flow_vis
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


def get_precoding_network():
    opt = {'type': 'PrecodingResNet', 'num_in_ch': 1, 'num_out_ch': 1, 'num_feat': 64}
    net = build_network(opt)
    return net


def get_singleframecompressor():
    opt = {'type': 'SingleFrameCompressor', 'num_ch': 1, 'search_size': 21, 'block_size': 4, 'color': 'Y'}
    net = build_network(opt)
    return net


def get_doubleframecompressor():
    opt = {'type': 'DoubleFrameCompressor', 'num_ch': 1, 'search_size': 21, 'block_size': 4, 'color': 'Y'}
    net = build_network(opt)
    return net


if __name__ == '__main__':

    device = torch.device('cpu')

    # net = get_precoding_network().to(device)
    # inp = torch.randn(1, 1, 128, 128)
    # out = net(inp)
    # print(out.shape)

    # net = get_compressor().to(device)
    #
    # img = cv2.imread('data/im2.png')
    # ori_img = bgr2ycbcr(img, y_only=True)  # [H, W]
    # plt.imshow(ori_img, cmap='gray')
    # plt.title('Original Image')
    # plt.show()
    #
    # ori_images = torch.from_numpy(ori_img).unsqueeze(dim=0).unsqueeze(dim=0).float()  # [B, 1, H, W]
    #
    # com_images, likelihoods = net(ori_images, qp=10)  # [B, 1, H, W]
    #
    # com_img = (com_images.detach().squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8).squeeze()  # [H, W]
    # plt.imshow(com_img, cmap='gray')
    # plt.title('Compressed Image')
    # plt.show()
    #
    # print(np.mean(np.abs(com_img - ori_img)))
    # plt.imshow(np.abs(com_img - ori_img), cmap='gray')
    # plt.title('Difference Image')
    # plt.show()


    # net = get_singleframecompressor().to(device)
    #
    # ori_img = cv2.imread('data/im2.png')[:, :, [2, 1, 0]] / 255.
    # plt.imshow(ori_img)
    # plt.title('Original Image')
    # plt.show()
    #
    # ori_images = torch.from_numpy(ori_img).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 3, H, W]
    #
    # com_images, likelihoods = net(ori_images, qp=30, training=False)  # [B, 3, H, W]
    #
    # com_img = ((torch.clip(com_images, 0, 1) * 255.).detach().squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8)  # [H, W, 3]
    # plt.imshow(com_img)
    # plt.title('Compressed Image')
    # plt.show()


    # net = get_doubleframecompressor().to(device)
    #
    # ori_img_1 = cv2.imread('data/im1.png')[:, :, [2, 1, 0]] / 255.
    # plt.imshow(ori_img_1)
    # plt.title('Original Image 1')
    # plt.show()
    #
    # ori_img_2 = cv2.imread('data/im2.png')[:, :, [2, 1, 0]] / 255.
    # plt.imshow(ori_img_2)
    # plt.title('Original Image 2')
    # plt.show()
    #
    # ori_images_1 = torch.from_numpy(ori_img_1).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 3, H, W]
    # ori_images_2 = torch.from_numpy(ori_img_2).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 3, H, W]
    #
    # com_images_2, flow, likelihoods = net(ori_images_2, ori_images_1, qp=40, training=False)  # [B, 3, H, W]
    #
    # com_img_2 = ((torch.clip(com_images_2, 0, 1) * 255.).detach().squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8)  # [H, W, 3]
    # plt.imshow(com_img_2)
    # plt.title('Compressed Image 2')
    # plt.show()
    #
    # flow = flow.squeeze(0).cpu().numpy()
    # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    # plt.imshow(flow_color)
    # plt.show()


    net = get_doubleframecompressor().to(device)

    ori_img_1 = cv2.imread('data/im1.png') / 255.
    ori_img_1 = bgr2ycbcr(ori_img_1.astype(np.float32), y_only=True)
    plt.imshow(ori_img_1, cmap='gray')
    plt.title('Original Image 1')
    plt.show()

    ori_img_2 = cv2.imread('data/im2.png') / 255.
    ori_img_2 = bgr2ycbcr(ori_img_2.astype(np.float32), y_only=True)
    plt.imshow(ori_img_2, cmap='gray')
    plt.title('Original Image 2')
    plt.show()

    ori_images_1 = torch.from_numpy(ori_img_1).unsqueeze(dim=0).unsqueeze(dim=0).float()  # [B, 1, H, W]
    ori_images_2 = torch.from_numpy(ori_img_2).unsqueeze(dim=0).unsqueeze(dim=0).float()  # [B, 1, H, W]

    com_images_2, flow, likelihoods, ori_res, com_res = \
        net(ori_images_2, ori_images_1, qp=30, training=False, beta=500)  # [B, 1, H, W]

    com_img_2 = ((torch.clip(com_images_2, 0, 1) * 255.).detach().squeeze().numpy()).round().astype(np.uint8)  # [H, W, 1]
    plt.imshow(com_img_2, cmap='gray')
    plt.title('Compressed Image 2')
    plt.show()

    # flow = flow.squeeze(0).cpu().numpy()
    # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    # plt.imshow(flow_color)
    # plt.show()

    ori_res_img = ori_res.detach().squeeze().numpy().round().astype(np.uint8)
    plt.imshow(ori_res_img, cmap='gray')
    plt.title('Original Residual Image')
    plt.show()

    com_res_img = com_res.detach().squeeze().numpy().round().astype(np.uint8)
    plt.imshow(com_res_img, cmap='gray')
    plt.title('Compressed Residual Image')
    plt.show()

    diff_img = (ori_res - com_res).detach().squeeze().numpy().round().astype(np.uint8)
    plt.imshow(np.abs(diff_img), cmap='gray')
    plt.title('Compressed Residual Image')
    plt.show()
