import os
import os.path as osp
import sys
import cv2
import glob
import json
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from shutil import get_terminal_size
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import mkdir
from IQA_pytorch import MS_SSIM


class ProgressBar(object):
    """A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


def compare_results():
    crf = 23
    src_root = f'/home/xiyang/Datasets/ReCp/results'
    dsr_root = f'/home/xiyang/Datasets/ReCp/compare_E-D_crf{crf}'
    res_path = f'/home/xiyang/Datasets/ReCp/results_E-D_crf{crf}.txt'

    mkdir(dsr_root)

    gt = 'GT'
    compare_list = [f'001_MSRResNet_BIC_x2_Vimeo90k_250k_crf{crf}/visualization/Vimeo90k_crf{crf}_frame',
                    f'001_MSRResNet_x2_f64b16_Vimeo90k_250k_B16G1_wandb/visualization/Vimeo90k_crf{crf}_frame']

    gt_img_paths = sorted(glob.glob(os.path.join(src_root, gt, '*', '*', 'im4.png')))

    avg_psnr_list = [0 for i in range(len(compare_list))]

    for gt_img_path in gt_img_paths:
        print(gt_img_path)
        tmp = gt_img_path.split('/')
        name_1, name_2, name_3 = tmp[-3], tmp[-2], tmp[-1]

        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
        src_img_path_list = [gt_img_path.replace(gt, compare_list[idx]) for idx in range(len(compare_list))]
        src_img_list = [cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED) for src_img_path in src_img_path_list]

        psnr_threshold = 60
        for i in range(len(compare_list)):
            psnr = min(calculate_psnr(img1=src_img_list[i], img2=gt_img, crop_border=False), psnr_threshold)
            avg_psnr_list[i] += psnr / len(gt_img_paths)
            print('PSNR {}: {:.2f} dB, Avg PSNR {}: {:.2f} dB'.format(i + 1, psnr, i + 1, avg_psnr_list[i]))
        if name_3.split('.')[0] == 'im4':
            src_img_list.append(gt_img)
            img = np.concatenate(tuple(src_img_list), axis=1)
            cv2.imwrite(osp.join(dsr_root, '{}_{}_{}.png'.format(name_1, name_2, name_3.split('.')[0])), img=img)

    with open(res_path, 'w') as f:
        for i in range(len(compare_list)):
            result = 'Avg PSNR {}: {:.2f} dB\n'.format(i + 1, avg_psnr_list[i])
            print(result)
            f.write(result)


def get_json(experiment_name):
    setting_names = ['Vimeo90k_crf18_video', 'Vimeo90k_crf23_video', 'Vimeo90k_crf28_video', 'Vimeo90k_crf33_video']
    for setting_name in setting_names:
        read_root = f'/{src_root}/{experiment_name}/visualization/{setting_name}'
        save_root = f'/{src_root}/{experiment_name}/visualization/{setting_name}_json'
        video_paths = sorted(glob.glob(osp.join(read_root, '*', '*.mp4')))
        for video_path in video_paths:
            print(video_path)
            names = video_path.split('/')
            mkdir(osp.join(save_root, names[-2]))
            json_path = osp.join(save_root, names[-2], names[-1].replace('.mp4', '.json'))
            command = 'ffmpeg_bitrate_stats -a gop -of json {} > {}'.format(video_path, json_path)
            os.system(command=command)


def get_bpp(src_root, experiment_name):
    setting_names = ['Vimeo90k_crf18_video', 'Vimeo90k_crf23_video', 'Vimeo90k_crf28_video', 'Vimeo90k_crf33_video']
    results_list = []
    H, W = 256, 448
    for setting_name in setting_names:
        root = f'/{src_root}/{experiment_name}/visualization/{setting_name}_json'
        json_paths = sorted(glob.glob(osp.join(root, '*', '*.json')))
        bpp_list = [0 for i in range(len(json_paths))]
        pbar = ProgressBar(len(json_paths))
        for idx, json_path in enumerate(json_paths):
            names = json_path.split('/')
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
            avg_bitrate = data['avg_bitrate']
            avg_fps = data['avg_fps']
            bpp_list[idx] = avg_bitrate / (H * W * avg_fps)
            pbar.update()
            # pbar.update('Processing {}/{}: {:.8f}'.format(names[-2], names[-1], bpp_list[idx]))
        print('{}, Avg BPP: {:.8f}'.format(setting_name, sum(bpp_list) / len(bpp_list)))
        results_list.append('{:.8f}'.format(sum(bpp_list) / len(bpp_list)))
    print(results_list)


def get_psnr(src_root, experiment_name):
    setting_names = ['Vimeo90k_crf18_frame', 'Vimeo90k_crf23_frame', 'Vimeo90k_crf28_frame', 'Vimeo90k_crf33_frame']
    results_list = []
    ref_img_paths = sorted(glob.glob(os.path.join(src_root, 'GT', '*', '*', 'im4.png')))
    for setting_name in setting_names:
        psnr_list = [0 for i in range(len(ref_img_paths))]
        pbar = ProgressBar(len(ref_img_paths))
        for idx, ref_img_path in enumerate(ref_img_paths):
            names = ref_img_path.split('/')
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)
            inp_img_path = ref_img_path.replace('/GT/', '/{}/visualization/{}/'.format(experiment_name, setting_name))
            inp_img = cv2.imread(inp_img_path, cv2.IMREAD_UNCHANGED)
            psnr_list[idx] = calculate_psnr(img1=inp_img, img2=ref_img, crop_border=False)
            pbar.update()
            # pbar.update('Processing {}/{}/{}: {:.4f} dB'.format(names[-3], names[-2], names[-1], psnr_list[idx]))
        print('{}, Avg PSNR: {:.4f} dB'.format(setting_name, sum(psnr_list) / len(psnr_list)))
        results_list.append('{:.4f}'.format(sum(psnr_list) / len(psnr_list)))
    print(results_list)


def get_msssim(src_root, experiment_name):
    setting_names = ['Vimeo90k_crf18_frame', 'Vimeo90k_crf23_frame', 'Vimeo90k_crf28_frame', 'Vimeo90k_crf33_frame']
    results_list = []
    ref_img_paths = sorted(glob.glob(os.path.join(src_root, 'GT', '*', '*', 'im4.png')))
    for setting_name in setting_names:
        msssim_list = [0 for i in range(len(ref_img_paths))]
        pbar = ProgressBar(len(ref_img_paths))
        for idx, ref_img_path in enumerate(ref_img_paths):
            names = ref_img_path.split('/')
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)
            inp_img_path = ref_img_path.replace('/GT/', '/{}/visualization/{}/'.format(experiment_name, setting_name))
            inp_img = cv2.imread(inp_img_path, cv2.IMREAD_UNCHANGED)
            ref_img = TF.to_tensor(ref_img[:, :, [2, 1, 0]]).unsqueeze(0)
            inp_img = TF.to_tensor(inp_img[:, :, [2, 1, 0]]).unsqueeze(0)
            ms_ssim = MS_SSIM(channels=3)
            msssim_list[idx] = ms_ssim(inp_img, ref_img, as_loss=False).item()
            pbar.update()
            pbar.update('Processing {}/{}/{}: {:.6f}'.format(names[-3], names[-2], names[-1], msssim_list[idx]))
        print('{}, Avg MS-SSIM: {:.6f}'.format(setting_name, sum(msssim_list) / len(msssim_list)))
        results_list.append('{:.6f}'.format(sum(msssim_list) / len(msssim_list)))
    print(results_list)


if __name__ == '__main__':

    src_root = f'/home/xiyang/Datasets/ReCp/results'

    experiment_name = '001_MSRResNet_x2_f64b16_Vimeo90k_250k_B16G1_wandb'
    # experiment_name = '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf23'
    # experiment_name = '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf28'
    # experiment_name = '001_MSRResNet_BIC_x2_Vimeo90k_250k_crf23'
    # experiment_name = '001_MSRResNet_BIC_x2_Vimeo90k_250k_crf28'

    # compare_results()
    # get_json()

    print(experiment_name)
    get_bpp(src_root, experiment_name)
    # get_psnr(src_root, experiment_name)
    # get_msssim(src_root, experiment_name)




