import os
import cv2
import glob
from basicsr.utils.matlab_functions import bgr2ycbcr, ycbcr2bgr
from resizer import imresize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def bgr_to_y_vimeo90k(src_root, dst_root):
    seq_paths = sorted(glob.glob(os.path.join(src_root, '*', '*', '*.png')))

    for src_img_path in seq_paths:
        print(src_img_path)
        tmp_list = src_img_path.split('/')
        name_a, name_b, img_name = tmp_list[-3], tmp_list[-2], tmp_list[-1]
        src_img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
        dst_img = bgr2ycbcr(src_img, y_only=True)
        mkdir(os.path.join(dst_root, name_a, name_b))
        cv2.imwrite(os.path.join(dst_root, name_a, name_b, img_name), dst_img)


def imresize_vimeo90k(src_root, dst_root, scale=0.5):
    seq_paths = sorted(glob.glob(os.path.join(src_root, '*', '*', '*.png')))

    for src_img_path in seq_paths:
        print(src_img_path)
        tmp_list = src_img_path.split('/')
        name_a, name_b, img_name = tmp_list[-3], tmp_list[-2], tmp_list[-1]
        src_img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
        dst_img = imresize(src_img, scale_factor=scale)
        mkdir(os.path.join(dst_root, name_a, name_b))
        cv2.imwrite(os.path.join(dst_root, name_a, name_b, img_name), dst_img)


if __name__ == '__main__':
    src_root = '/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_img_SRx2_crf18_img/sequences'
    dst_root = '/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_img_SRx2_crf18_img_y/sequences'
    bgr_to_y_vimeo90k(src_root, dst_root)
    # imresize_vimeo90k(src_root, dst_root)

    # img = cv2.imread('/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_y/sequences/00001/0001/im4.png', cv2.IMREAD_UNCHANGED)
