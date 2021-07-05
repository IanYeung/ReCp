import cv2
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchjpeg.dct import blockify, deblockify, block_dct, block_idct, batch_dct, batch_idct, to_rgb, to_ycbcr
from basicsr.archs.arch_util import pixel_unshuffle
from basicsr.utils.matlab_functions import ycbcr2rgb, rgb2ycbcr, bgr2ycbcr, ycbcr2bgr

C_f4 = np.array([[1,  1,  1,  1],
                 [2,  1, -1, -2],
                 [1, -1, -1,  1],
                 [1, -2,  2, -1]],
                dtype=np.float32)

C_i4 = np.array([[  1,   1,    1,    1],
                 [  1, 1/2, -1/2,   -1],
                 [  1,  -1,   -1,    1],
                 [1/2,  -1,    1, -1/2]],
                dtype=np.float32)

S_f4 = np.array([[1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                 [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10],
                 [1/4, 1/(2 * math.sqrt(10)), 1/4, 1/(2 * math.sqrt(10))],
                 [1/(2 * math.sqrt(10)), 1/10, 1/(2 * math.sqrt(10)), 1/10]],
                dtype=np.float32)

S_i4 = np.array([[1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                 [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5],
                 [1/4, 1/math.sqrt(10), 1/4, 1/math.sqrt(10)],
                 [1/math.sqrt(10), 2/5, 1/math.sqrt(10), 2/5]],
                dtype=np.float32)

Q_step_table = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125]
# Q_step_table = [0.625, 0.7031, 0.7812, 0.8984, 0.9766, 1.1328]
# Q_step_table = [0.6423, 0.6917, 0.7906, 0.8894, 0.9882, 1.1364]


if __name__ == '__main__':

    def V_i4(qp, return_type='tensor'):
        v = np.array([[10, 16, 13],
                      [11, 18, 14],
                      [13, 20, 16],
                      [14, 23, 18],
                      [16, 25, 20],
                      [18, 29, 23]], dtype=np.float32)
        if return_type == 'numpy':
            return np.array([[v[qp, 0], v[qp, 2], v[qp, 0], v[qp, 2]],
                             [v[qp, 2], v[qp, 1], v[qp, 2], v[qp, 1]],
                             [v[qp, 0], v[qp, 2], v[qp, 0], v[qp, 2]],
                             [v[qp, 2], v[qp, 1], v[qp, 2], v[qp, 1]]],
                            dtype=np.float32)
        else:
            return torch.from_numpy(
                np.array([[v[qp, 0], v[qp, 2], v[qp, 0], v[qp, 2]],
                          [v[qp, 2], v[qp, 1], v[qp, 2], v[qp, 1]],
                          [v[qp, 0], v[qp, 2], v[qp, 0], v[qp, 2]],
                          [v[qp, 2], v[qp, 1], v[qp, 2], v[qp, 1]]],
                         dtype=np.float32)
            )

    def M_f4(qp, return_type='tensor'):
        m = np.array([[13107, 5243, 8066],
                      [11916, 4660, 7490],
                      [10082, 4194, 6554],
                      [9362, 3647, 5825],
                      [8192, 3355, 5243],
                      [7282, 2893, 4559]], dtype=np.float32)
        if return_type == 'numpy':
            return np.array([[m[qp, 0], m[qp, 2], m[qp, 0], m[qp, 2]],
                             [m[qp, 2], m[qp, 1], m[qp, 2], m[qp, 1]],
                             [m[qp, 0], m[qp, 2], m[qp, 0], m[qp, 2]],
                             [m[qp, 2], m[qp, 1], m[qp, 2], m[qp, 1]]],
                            dtype=np.float32)
        else:
            return torch.from_numpy(
                np.array([[m[qp, 0], m[qp, 2], m[qp, 0], m[qp, 2]],
                          [m[qp, 2], m[qp, 1], m[qp, 2], m[qp, 1]],
                          [m[qp, 0], m[qp, 2], m[qp, 0], m[qp, 2]],
                          [m[qp, 2], m[qp, 1], m[qp, 2], m[qp, 1]]],
                         dtype=np.float32)
            )

    # qp_list = [0, 1, 2, 3, 4, 5]
    # for qp in qp_list:
    #     print('-' * 20, f'V Matrix, QP: {qp}', '-' * 20)
    #     print(get_V_i4(qp))
    #
    # for qp in qp_list:
    #     print('-' * 20, f'M Matrix, QP: {qp}', '-' * 20)
    #     print(get_M_f4(qp))

    # qp = 0
    #
    # C_f4 = torch.from_numpy(C_f4)
    # C_i4 = torch.from_numpy(C_i4)
    # S_f4 = torch.from_numpy(S_f4)
    # S_i4 = torch.from_numpy(S_i4)
    #
    # X = np.array([[58, 64, 51, 58],
    #               [52, 64, 56, 66],
    #               [62, 63, 61, 64],
    #               [59, 51, 63, 69]], dtype=np.float32)
    # X = torch.from_numpy(X)
    # Y = C_f4 @ X @ C_f4.T * S_f4
    # print(Y)
    #
    # Y_h = torch.round(Y / Q_step_table[qp % 6])
    # print(Y_h)
    # Y_s = Y_h * Q_step_table[qp % 6]
    # print(Y_s)
    #
    # Z = torch.round(C_i4.T @ (Y_s * S_i4) @ C_i4)
    # print(Z)
    #
    # print(C_f4 @ X @ C_f4.T * M_f4(qp=(qp % 6)))
    # print(C_f4 @ X @ C_f4.T * M_f4(qp=(qp % 6)) / (2 ** (15 + math.floor(qp / 6))))
    # print(torch.round(Y * m / (2 ** (15 + math.floor(qp / 6)))))
    #
    # def f_transform(X, C_f4, M_f4, qp):
    #     return torch.round(C_f4 @ X @ C_f4.T * M_f4(qp=(qp % 6)) / (2 ** (15 + math.floor(qp / 6))))
    #
    # Y = f_transform(X, C_f4, M_f4, qp=qp)
    # print(Y)
    #
    # def i_transform(Y, C_i4, V_i4, qp):
    #     return torch.round(C_i4.T @ (Y * V_i4(qp=(qp % 6)) * (2 ** (math.floor(qp / 6)))) @ C_i4 / (2 ** 6))
    #
    # Z = i_transform(b, C_i4, V_i4, qp=qp)
    # print(Z)

    qp = 0

    C_f4 = torch.from_numpy(C_f4)
    C_i4 = torch.from_numpy(C_i4)
    S_f4 = torch.from_numpy(S_f4)
    S_i4 = torch.from_numpy(S_i4)

    block_size = 4
    img = cv2.imread('/home/xiyang/Downloads/0100/im1.png')
    ori_img = bgr2ycbcr(img, y_only=True)  # [H, W]
    plt.imshow(ori_img, cmap='gray')
    plt.title('Original Image')
    plt.show()

    ori_images = torch.from_numpy(ori_img).unsqueeze(dim=0).unsqueeze(dim=0).float()  # [B, 1, H, W]

    X = blockify(ori_images, size=block_size)  # [B, 1, (H/bH)*(W/bW), bH, bW]

    C_f4, S_f4 = C_f4.expand(*X.shape), S_f4.expand(*X.shape)

    # forward transform
    Y = C_f4 @ X @ C_f4.transpose(-1, -2) * S_f4

    # quantization
    Q_step = Q_step_table[qp % 6] * (2 ** math.floor(qp / 6))
    Y_h = torch.round(Y / Q_step) * Q_step

    # backward transform
    Z = torch.round(C_i4.transpose(-1, -2) @ (Y_h * S_i4) @ C_i4)

    com_images = deblockify(Z, size=(256, 448))  # [B, 1, H, W]

    com_img = (com_images.squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8).squeeze()  # [H, W]
    plt.imshow(com_img, cmap='gray')
    plt.title('Compressed Image')
    plt.show()

    print(np.mean(np.abs(com_img - ori_img)))
    plt.imshow(np.abs(com_img - ori_img), cmap='gray')
    plt.title('Difference Image')
    plt.show()

    # extract DCT sub-band
    dct_subband_list_v1 = []
    for i in range(4):
        for j in range(4):
            Y_ij = Y[:, :, :, i:i + 1, j:j + 1]
            dct_subband = deblockify(Y_ij, size=(256 // 4, 448 // 4))
            dct_subband_list_v1.append(dct_subband.squeeze())
            dct_subband = (dct_subband.squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8).squeeze()
            # dct_subband_list.append(torch.from_numpy(dct_subband))
            # plt.imshow(dct_subband, cmap='gray')
            # plt.title(f'Sub-band i:{i},j:{j}')
            # plt.savefig(f'/home/xiyang/Results/ReCp/dct_subband_i{i}j{j}.png')
            cv2.imwrite(f'/home/xiyang/Results/ReCp/dct_subband_i{i}j{j}.png', img=dct_subband)

    dct_subband_list_v2 = []
    Y_sub = pixel_unshuffle(deblockify(Y, size=(256, 448)), scale=4)
    for i in range(4 * 4):
        dct_subband_list_v2.append(Y_sub[:, i:i + 1, :, :].squeeze())

    # reconstruct_list = []
    # for i in range(4):
    #     for j in range(4):
    #         Y_ij = torch.zeros_like(Y)
    #         Y_ij[:, :, :, i:i+1, j:j+1] = Y[:, :, :, i:i+1, j:j+1]
    #         Z_ij = torch.round(C_i4.transpose(-1, -2) @ (Y_ij * S_i4) @ C_i4)
    #         reconstruct = deblockify(Z_ij, size=(256, 448))
    #         reconstruct_list.append(reconstruct.squeeze())
    #         reconstruct = (reconstruct.squeeze(dim=0).permute(1, 2, 0).numpy()).round().astype(np.uint8).squeeze()
    #         # reconstruct_list.append(torch.from_numpy(reconstruct))
    #         # plt.imshow(reconstruct, cmap='gray')
    #         # plt.title(f'Sub-band i:{i},j:{j}')
    #         # plt.savefig(f'/home/xiyang/Results/ReCp/reconstruct_i{i}j{j}.png')
    #         cv2.imwrite(f'/home/xiyang/Results/ReCp/reconstruct_i{i}j{j}.png', img=reconstruct)
