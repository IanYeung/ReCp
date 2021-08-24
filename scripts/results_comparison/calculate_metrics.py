import os
import sys
import cv2
import glob
import json
import time
import shlex
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from shutil import get_terminal_size
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import mkdir

from ffmpeg_quality_metrics import FfmpegQualityMetrics as ffqm


def calculate_metrics():
    src_video_root = '/home/xiyang/Datasets/MCL-JVC/360P-MP4-SR-CRF28'
    dst_video_root = '/home/xiyang/Datasets/MCL-JVC/720P-MP4'

    # src_video_path = os.path.join(src_video_root, 'videoSRC01_1280x720_30_x2.mp4')
    # dst_video_path = os.path.join(dst_video_root, 'videoSRC01_1280x720_30.mp4')

    src_video_paths = sorted(glob.glob(os.path.join(src_video_root, '*')))
    dst_video_paths = sorted(glob.glob(os.path.join(dst_video_root, '*')))

    assert len(src_video_paths) == len(dst_video_paths)

    avg_psnr_y, avg_ssim_y, avg_vmaf = 0., 0., 0.

    for idx, (src_video_path, dst_video_path) in enumerate(zip(src_video_paths, dst_video_paths)):
        # print('src path: {}'.format(src_video_path))
        # print('dst path: {}'.format(dst_video_path))

        results = ffqm(src_video_path, dst_video_path).calc(['ssim', 'psnr', 'vmaf'])

        psnr_y_list = [frm_results['psnr_y'] for frm_results in results['psnr']]
        ssim_y_list = [frm_results['ssim_y'] for frm_results in results['ssim']]
        vmaf_list = [frm_results['vmaf'] for frm_results in results['vmaf']]

        seq_avg_psnr_y = sum(psnr_y_list) / len(psnr_y_list)
        seq_avg_ssim_y = sum(ssim_y_list) / len(ssim_y_list)
        seq_avg_vmaf = sum(vmaf_list) / len(vmaf_list)

        avg_psnr_y += seq_avg_psnr_y / len(src_video_paths)
        avg_ssim_y += seq_avg_ssim_y / len(src_video_paths)
        avg_vmaf += seq_avg_vmaf / len(src_video_paths)

        print('Idx {:02d}:\t PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'
              .format(idx + 1, seq_avg_psnr_y, seq_avg_ssim_y, seq_avg_vmaf))

    print('---------------' * 4)
    print('PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'.format(avg_psnr_y, avg_ssim_y, avg_vmaf))


def getVideoMetadata(pathToInputVideo):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(pathToInputVideo)

    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # # prints all the metadata available:
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(ffprobeOutput)
    #
    # h = ffprobeOutput['streams'][0]['height']
    # w = ffprobeOutput['streams'][0]['width']

    kbps = float(ffprobeOutput['streams'][0]['bit_rate']) / 1000

    return kbps


def get_bitrate(src_root='/home/xiyang/Datasets/MCL-JVC/720P-MP4'):
    src_video_paths = sorted(glob.glob(os.path.join(src_root, '*.mp4')))
    avg_kbps = 0.
    for src_video_path in src_video_paths:
        src_video_name = os.path.basename(src_video_path)
        kbps = getVideoMetadata(pathToInputVideo=src_video_path)
        avg_kbps += kbps
        print('{}:\t {:.3f} kbps'.format(src_video_name, kbps))
    print('---------------' * 2)
    print('Average bitrate {:.3f} kbps'.format(avg_kbps / len(src_video_paths)))


def save_bitrate_info(src_root='/home/xiyang/Datasets/MCL-JVC/720P-MP4',
                      dst_root='/home/xiyang/Datasets/MCL-JVC/720P-JSON'):
    mkdir(dst_root)
    src_video_paths = sorted(glob.glob(os.path.join(src_root, '*.mp4')))
    for src_video_path in src_video_paths:
        src_video_name = os.path.basename(src_video_path)
        dst_json_path = os.path.join(dst_root, src_video_name.replace('.mp4', '.json'))
        command = 'ffmpeg_bitrate_stats -a gop -of json {} > {}'.format(src_video_path, dst_json_path)
        os.system(command=command)


if __name__ == '__main__':
    # calculate_metrics()
    get_bitrate()
    # save_bitrate_info()
