import os
import glob


def rename(root):
    img_paths = sorted(glob.glob(os.path.join(root, '*', '*', '*.png')))
    for ori_img_path in img_paths:
        dst_img_path = ori_img_path.replace('im1.png', 'im4.png')
        print('rename {} into {}'.format(ori_img_path, dst_img_path))
        os.rename(ori_img_path, dst_img_path)


def remove(root):
    img_paths = sorted(glob.glob(os.path.join(root, '*', '*', 'im4.png')))
    for img_path in img_paths:
        print('remove {}'.format(img_path))
        os.remove(img_path)


if __name__ == '__main__':
    mode = 'crf'
    # exp_name = f'001_MSRResNet_x2_f64b16_Vimeo90k_250k_B16G1_wandb'
    # exp_name = f'001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf23'
    exp_name = f'001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf28'
    # exp_name = f'001_MSRResNet_BIC_x2_Vimeo90k_250k_crf23'
    # exp_name = f'001_MSRResNet_BIC_x2_Vimeo90k_250k_crf28'
    for quality in [18, 23, 28, 33]:
        root = f'/home/xiyang/Datasets/ReCp/results/{exp_name}/visualization/Vimeo90k_{mode}{quality}_frame'
        rename(root)
