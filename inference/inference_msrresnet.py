import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.utils import mkdir
from basicsr.utils.matlab_functions import ycbcr2bgr, bgr2ycbcr, imresize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/yangxi/projects/ReCp/experiments/pretrained_models/MSRResNet_x2_JointTrain.pth'
    )
    parser.add_argument(
        '--src_root',
        type=str,
        default='/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_img/sequences'
    )
    parser.add_argument(
        '--dst_root',
        type=str,
        default='/home/yangxi/projects/ReCp/results/MSRResNet_x2_Joint_Vimeo90k_250k'
    )
    parser.add_argument(
        '--inference_list',
        type=str,
        default='/home/yangxi/projects/ReCp/basicsr/data/meta_info/meta_info_Vimeo90K_test_GT.txt'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:1'
    )
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=2)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    with open(args.inference_list, 'r') as fin:
        sequence_list = [line.split(' ')[0] for line in fin]

    for sequence in sequence_list:
        name_1, name_2 = sequence.split('/')
        img_paths = sorted(glob.glob(os.path.join(args.src_root, name_1, name_2, '*')))
        for img_path in img_paths:

            imgname = os.path.splitext(os.path.basename(img_path))[0]
            print(f'Testing {name_1}/{name_2}/{imgname}')
            # read image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # inference
            with torch.no_grad():
                output = model(img)
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            mkdir(os.path.join(args.dst_root, name_1, name_2))
            cv2.imwrite(os.path.join(args.dst_root, name_1, name_2, f'{imgname}.png'), output)

            # imgname = os.path.splitext(os.path.basename(img_path))[0]
            # print(f'Testing {name_1}/{name_2}/{imgname}')
            # # read image
            # img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            # img_ycbcr = bgr2ycbcr(img_bgr)
            # img_y = img_ycbcr[:, :, 0:1]
            # img_c = img_ycbcr[:, :, 1:3]
            # # inference
            # img_y = torch.from_numpy(np.transpose(img_y, (2, 0, 1))).float()
            # img_y = img_y.unsqueeze(0).to(device)  # [1, 1, H, W]
            # with torch.no_grad():
            #     out_y = model(img_y)
            # out_y = out_y.data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
            # out_y = np.transpose(out_y, (1, 2, 0))
            # out_c = imresize(img_c, scale=2)
            # out_ycbcr = np.concatenate([out_y, out_c], axis=2)
            # out_bgr = ycbcr2bgr(out_ycbcr)
            # out_bgr = (out_bgr * 255.0).round().astype(np.uint8)
            # # save image
            # mkdir(os.path.join(args.dst_root, name_1, name_2))
            # cv2.imwrite(os.path.join(args.dst_root, name_1, name_2, f'{imgname}.png'), out_bgr)


if __name__ == '__main__':
    main()
