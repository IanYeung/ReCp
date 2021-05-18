import os
import glob


def rename(root):
    img_paths = sorted(glob.glob(os.path.join(root, '*', '*', '*.png')))
    for ori_img_path in img_paths:
        dst_img_path = ori_img_path.replace('im1.png', 'im4.png')
        command = 'mv {} {}'.format(ori_img_path, dst_img_path)
        print(command)
        os.system(command=command)


if __name__ == '__main__':
    root = '/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_com_crf28_img/sequences'
    rename(root)