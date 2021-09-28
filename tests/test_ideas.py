import os
import cv2
import math
import flow_vis

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import functional as TF
from torchjpeg.dct import blockify, deblockify, block_dct, block_idct, batch_dct, batch_idct, to_rgb, to_ycbcr

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

from basicsr.utils.matlab_functions import bgr2ycbcr, ycbcr2bgr
from basicsr.archs.arch_util import Prediction, Rescaler


def forward(curr_frame, prev_frame, qp, training=False, mode='inter', color='Y'):
    rescaler = Rescaler()

    if mode == 'inter':
        if color == 'RGB':
            predicted, flow = Prediction.inter_prediction_ste_rgb(curr_frame, prev_frame,
                                                                  search_size=21,
                                                                  block_size=4)
        else:
            predicted, flow = Prediction.inter_prediction_ste(curr_frame, prev_frame,
                                                              search_size=21,
                                                              block_size=4)
    else:
        if color == 'RGB':
            predicted, flow = Prediction.intra_prediction_ste_rgb(curr_frame, curr_frame,
                                                                  search_size=21,
                                                                  block_size=4)
        else:
            predicted, flow = Prediction.intra_prediction_ste(curr_frame, curr_frame,
                                                              search_size=21,
                                                              block_size=4)
    residual = curr_frame - predicted
    residual = rescaler.fwd_rescale_pt(residual) * 255.

    residual = forward_residual(residual, qp, training)

    residual = rescaler.bwd_rescale_pt(residual / 255.)
    curr_frame = predicted + residual

    return curr_frame, flow


def forward_residual(ori_images, qp):

    C_f4 = torch.tensor([[1, 1, 1, 1],
                         [2, 1, -1, -2],
                         [1, -1, -1, 1],
                         [1, -2, 2, -1]],
                        dtype=torch.float32)

    C_i4 = torch.tensor([[1, 1, 1, 1],
                         [1, 1 / 2, -1 / 2, -1],
                         [1, -1, -1, 1],
                         [1 / 2, -1, 1, -1 / 2]],
                        dtype=torch.float32)

    S_f4 = torch.tensor([[1 / 4, 1 / (2 * math.sqrt(10)), 1 / 4, 1 / (2 * math.sqrt(10))],
                         [1 / (2 * math.sqrt(10)), 1 / 10, 1 / (2 * math.sqrt(10)), 1 / 10],
                         [1 / 4, 1 / (2 * math.sqrt(10)), 1 / 4, 1 / (2 * math.sqrt(10))],
                         [1 / (2 * math.sqrt(10)), 1 / 10, 1 / (2 * math.sqrt(10)), 1 / 10]],
                        dtype=torch.float32)

    S_i4 = torch.tensor([[1 / 4, 1 / math.sqrt(10), 1 / 4, 1 / math.sqrt(10)],
                         [1 / math.sqrt(10), 2 / 5, 1 / math.sqrt(10), 2 / 5],
                         [1 / 4, 1 / math.sqrt(10), 1 / 4, 1 / math.sqrt(10)],
                         [1 / math.sqrt(10), 2 / 5, 1 / math.sqrt(10), 2 / 5]],
                        dtype=torch.float32)

    Q_step_table = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125]

    B, C, H, W = ori_images.shape

    X = blockify(ori_images, size=4)
    C_f4, S_f4 = C_f4.expand(*X.shape), S_f4.expand(*X.shape)

    # forward transform
    Y = C_f4 @ X @ C_f4.transpose(-1, -2) * S_f4

    # quantization
    Q_step = Q_step_table[qp % 6] * (2 ** math.floor(qp / 6))
    Y_h = Y / Q_step

    dct_subbands_list = extract_subband(Y_h, H, W)
    dct_subbands_list = list(map(torch.round, dct_subbands_list))
    Y_h = combine_subband(dct_subbands_list)

    Y_h = Y_h * Q_step

    # backward transform
    Z = torch.round(C_i4.transpose(-1, -2) @ (Y_h * S_i4) @ C_i4)

    com_images = deblockify(Z, size=(H, W))  # [B, C, H, W]
    return com_images, dct_subbands_list


def extract_subband(Y_h, H, W):
    dct_subband_list = []
    Y_h = deblockify(Y_h, size=(H, W))  # [B, C, H, W]
    Y_sub = F.pixel_unshuffle(Y_h, downscale_factor=4)  # [B, C, H//4, W//4]
    for i in range(4 * 4):
        dct_subband_list.append(Y_sub[:, (i) * Y_h.size(1):(i + 1) * Y_h.size(1), :, :])
    return dct_subband_list


def combine_subband(dct_subband_list):
    dct_subband = torch.cat(dct_subband_list, dim=1)  # [B, C, H//4, W//4]
    Y_h = F.pixel_shuffle(dct_subband, upscale_factor=4)  # [B, C, H, W]
    Y_h = blockify(Y_h, size=4)
    return Y_h


def visualize_prediction():
    mode = 'intra'
    save_root = '/home/xiyang/Results/ReCp/prediction'

    device = torch.device('cpu')

    im1 = cv2.imread(filename='data/im1.png', flags=cv2.IMREAD_UNCHANGED) / 255.
    im2 = cv2.imread(filename='data/im2.png', flags=cv2.IMREAD_UNCHANGED) / 255.

    # # RGB mode
    # im1 = im1.astype(np.float32)
    # im2 = im2.astype(np.float32)
    # im1_tensor = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0).to(device)
    # im2_tensor = torch.from_numpy(im2).permute(2, 0, 1).unsqueeze(0).to(device)
    #
    # out_tensor = Prediction.intra_prediction_rgb(im1_tensor, im2_tensor, search_size=21, block_size=4, soft=False)
    # out_tensor = Prediction.inter_prediction_rgb(im1_tensor, im2_tensor, search_size=21, block_size=4, soft=False)
    # if mode == 'intra':
    #     out_tensor, flow_tensor = Prediction.intra_prediction_ste_rgb(im1_tensor, im2_tensor, search_size=21, block_size=4)
    # elif mode == 'inter':
    #     out_tensor, flow_tensor = Prediction.inter_prediction_ste_rgb(im1_tensor, im2_tensor, search_size=21, block_size=4)
    # else:
    #     raise ValueError()
    #
    # out = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # plt.imshow(out[:, :, [2, 1, 0]])
    # plt.title('Predicted')
    # plt.show()
    # plt.savefig(os.path.join(save_root, '{}_predicted.png'.format(mode)))
    #
    # plt.imshow(im1[:, :, [2, 1, 0]])
    # plt.title('Original')
    # plt.show()
    # plt.savefig(os.path.join(save_root, '{}_original.png'.format(mode)))
    #
    # res = im1 - out
    # plt.imshow(np.abs(res[:, :, [2, 1, 0]]))
    # plt.title('Residual')
    # plt.show()
    # plt.savefig(os.path.join(save_root, '{}_residual.png'.format(mode)))
    #
    # flow = flow_tensor.squeeze(0).cpu().numpy()
    # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    # plt.imshow(flow_color)
    # plt.title('Flow')
    # plt.show()
    # plt.savefig(os.path.join(save_root, '{}_flow.png'.format(mode)))

    # Y mode
    im1 = bgr2ycbcr(im1.astype(np.float32), y_only=True)
    im2 = bgr2ycbcr(im2.astype(np.float32), y_only=True)
    im1_tensor = torch.from_numpy(im1).unsqueeze(0).unsqueeze(0).to(device)
    im2_tensor = torch.from_numpy(im2).unsqueeze(0).unsqueeze(0).to(device)

    # out_tensor = Prediction.intra_prediction(im1_tensor, im2_tensor, search_size=21, block_size=4, soft=False)
    # out_tensor = Prediction.inter_prediction(im1_tensor, im2_tensor, search_size=21, block_size=4, soft=False)
    if mode == 'intra':
        out_tensor, flow_tensor = Prediction.intra_prediction_ste(im1_tensor, im2_tensor, search_size=21, block_size=4)
    elif mode == 'inter':
        out_tensor, flow_tensor = Prediction.inter_prediction_ste(im1_tensor, im2_tensor, search_size=21, block_size=4)
    else:
        raise ValueError()
    out = out_tensor.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(out, cmap='gray')
    plt.title('Predicted')
    # plt.show()
    plt.savefig(os.path.join(save_root, '{}_predicted.png'.format(mode)))

    plt.imshow(im1, cmap='gray')
    plt.title('Original')
    # plt.show()
    plt.savefig(os.path.join(save_root, '{}_original.png'.format(mode)))

    res = im1 - out
    plt.imshow(np.abs(res), cmap='gray')
    plt.title('Residual')
    # plt.show()
    plt.savefig(os.path.join(save_root, '{}_residual.png'.format(mode)))

    flow = flow_tensor.squeeze(0).cpu().numpy()
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.title('Flow')
    # plt.show()
    plt.savefig(os.path.join(save_root, '{}_flow.png'.format(mode)))


def visualize_dctsubband():
    save_root = '/home/xiyang/Results/ReCp/dctsubband'

    ori_img_1 = cv2.imread('data/im1.png') / 255.
    ori_img_2 = cv2.imread('data/im2.png') / 255.

    ori_img_1 = np.expand_dims(bgr2ycbcr(ori_img_1.astype(np.float32), y_only=True), axis=-1)
    ori_img_2 = np.expand_dims(bgr2ycbcr(ori_img_2.astype(np.float32), y_only=True), axis=-1)
    # plt.imshow(ori_img_1, cmap='gray')
    # plt.title('Previous Frame (Original)')
    # plt.show()
    # plt.imshow(ori_img_2, cmap='gray')
    # plt.title('Current Frame (Original)')
    # plt.show()

    prev_frame = torch.from_numpy(ori_img_1).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 1, H, W]
    curr_frame = torch.from_numpy(ori_img_2).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 1, H, W]

    rescaler = Rescaler()

    predicted, flow = Prediction.inter_prediction_ste(curr_frame, prev_frame, search_size=21, block_size=4)

    residual = curr_frame - predicted
    residual = rescaler.fwd_rescale_pt(residual) * 255.

    residual, subbands = forward_residual(residual, qp=20)

    residual = rescaler.bwd_rescale_pt(residual / 255.)
    reco_frame = predicted + residual
    # plt.imshow(reco_frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    # plt.title('Current Frame (Compressed)')
    # plt.show()

    for idx, subband in enumerate(subbands):
        cv2.imwrite(
            filename=os.path.join(save_root, 'subband_{}_{}.png'.format(idx // 4, idx % 4)),
            img=subband.squeeze().cpu().numpy().round().astype(np.uint8)
        )


def visualize_basis_function():
    a = 1/2
    b = math.sqrt(1/2) * math.cos(math.pi * 1 / 8)
    c = math.sqrt(1/2) * math.cos(math.pi * 3 / 8)
    A = np.array([[a,  a,  a,  a],
                  [b,  c, -c, -b],
                  [a, -a, -a,  a],
                  [c, -b,  b, -c]],
                 dtype=np.float32)
    basis_image_list = []
    for i in range(4):
        for j in range(4):
            basis_image = np.outer(A[i, :], A.T[:, j])
            basis_image_list.append(torch.from_numpy(basis_image))
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.axis('off')
            plt.imshow(basis_image, cmap='gray', vmin=-1, vmax=1)
            print(f'----------i:{i}, j:{j}----------')
            print(basis_image)
    # plt.show()

    # Divisive Normalization
    fi = basis_image_list[0]
    fj = basis_image_list[1]
    sigma_fi = (1/6) * torch.linalg.vector_norm(fi, ord=1) + 0.05
    h_ij = torch.exp(-1 * (torch.linalg.vector_norm(fi - fj, ord=1) ** 2) / sigma_fi ** 2)


if __name__ == '__main__':
    # visualize_prediction()
    # visualize_dctsubband()


    a = 1 / 2
    b = math.sqrt(1 / 2) * math.cos(math.pi * 1 / 8)
    c = math.sqrt(1 / 2) * math.cos(math.pi * 3 / 8)
    A = np.array([[a, a, a, a],
                  [b, c, -c, -b],
                  [a, -a, -a, a],
                  [c, -b, b, -c]],
                 dtype=np.float32)
    basis_image_list = []
    for i in range(4):
        for j in range(4):
            basis_image = np.outer(A[i, :], A.T[:, j])
            basis_image_list.append(torch.from_numpy(basis_image))
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.axis('off')
            plt.imshow(basis_image, cmap='gray', vmin=-1, vmax=1)
            print(f'----------i:{i}, j:{j}----------')
            print(basis_image)
    plt.show()

    # Divisive Normalization
    H_ij = np.zeros((4*4, 4*4))
    for i in range(4*4):
        for j in range(4*4):
            fi = basis_image_list[i]
            fj = basis_image_list[j]
            sigma_fi = (1 / 6) * torch.linalg.vector_norm(fi, ord=1) + 0.05
            h_ij = torch.exp(-1 * (torch.linalg.vector_norm(fi - fj, ord=1) ** 2) / sigma_fi ** 2)
            H_ij[i, j] = h_ij


    # save_root = '/home/xiyang/Results/ReCp/dctsubband'
    #
    # ori_img_1 = cv2.imread('data/im1.png') / 255.
    # ori_img_2 = cv2.imread('data/im2.png') / 255.
    #
    # ori_img_1 = np.expand_dims(bgr2ycbcr(ori_img_1.astype(np.float32), y_only=True), axis=-1)
    # ori_img_2 = np.expand_dims(bgr2ycbcr(ori_img_2.astype(np.float32), y_only=True), axis=-1)
    # # plt.imshow(ori_img_1, cmap='gray')
    # # plt.title('Previous Frame (Original)')
    # # plt.show()
    # # plt.imshow(ori_img_2, cmap='gray')
    # # plt.title('Current Frame (Original)')
    # # plt.show()
    #
    # prev_frame = torch.from_numpy(ori_img_1).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 1, H, W]
    # curr_frame = torch.from_numpy(ori_img_2).permute(2, 0, 1).unsqueeze(dim=0).float()  # [B, 1, H, W]
    #
    # rescaler = Rescaler()
    #
    # predicted, flow = Prediction.inter_prediction_ste_y(curr_frame, prev_frame, search_size=21, block_size=4)
    #
    # residual = curr_frame - predicted
    # residual = rescaler.fwd_rescale_pt(residual) * 255.
    #
    # residual, subbands = forward_residual(residual, qp=20)
    #
    # residual = rescaler.bwd_rescale_pt(residual / 255.)
    # reco_frame = predicted + residual
    # # plt.imshow(reco_frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    # # plt.title('Current Frame (Compressed)')
    # # plt.show()
    #
    # H, W = 256 // 4, 448 // 4
    # beta = torch.tensor([0.01] * 16)
    # beta_expanded = beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
    # gamma = 0.98
    #
    # # [B, 16, H, W]
    # ci = torch.cat(subbands, dim=1)
    # # [B, 16, H, W]
    # ri = (torch.sgn(ci) * (torch.abs(ci) ** gamma)) / (beta_expanded + (torch.abs(ci) ** gamma))
    #
    # for idx in range(16):
    #     cv2.imwrite(
    #         filename=os.path.join(save_root, 'normalized_subband_{}_{}.png'.format(idx // 4, idx % 4)),
    #         img=ri.squeeze(0)[idx, :, :].cpu().numpy().round().astype(np.uint8)
    #     )
    #
    # # [B, H, W, 16]
    # ro = ri.permute(0, 2, 3, 1)
    # beta_expanded = beta_expanded.permute(0, 2, 3, 1)
    # identity = torch.eye(16).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, H, W, 1, 1)
    # d_a = torch.linalg.inv((identity - torch.diag_embed(ro)).view(-1, 16, 16))
    # d_b = torch.diag_embed(beta_expanded).view(-1, 16, 16)
    # d_c = torch.abs(ro).view(-1, 16, 1)
    # result = d_a @ d_b @ d_c
    # co = torch.sgn(ro) * (result.view(1, H, W, 16) ** (1 / gamma))
    # co = co.permute(0, 3, 1, 2)
    # print(torch.dist(ci, co))
    #
    # for idx in range(16):
    #     cv2.imwrite(
    #         filename=os.path.join(save_root, 'denormalized_subband_{}_{}.png'.format(idx // 4, idx % 4)),
    #         img=co.squeeze(0)[idx, :, :].cpu().numpy().round().astype(np.uint8)
    #     )
