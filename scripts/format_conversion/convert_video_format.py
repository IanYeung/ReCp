import os
import os.path as osp
import sys
import glob
from basicsr.utils import mkdir


def yuv_to_y4m(src_path, dst_path):
    w, h = 1920, 1080
    r = 25
    pix_fmt = 'yuv420p'

    src_videos = sorted(glob.glob(osp.join(src_path, '*.yuv')))
    for src_video in src_videos:
        info = osp.basename(src_video).split('.')[0].split('_')
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.yuv', '.y4m'))
        command = f'ffmpeg -s {info[-2]} -r {info[-1]} -pix_fmt {pix_fmt} -i {src_video} {dst_video} -y'
        print(command)
        os.system(command)


def y4m_to_h264(src_path, dst_path, crf=22):
    src_videos = sorted(glob.glob(osp.join(src_path, '*.y4m')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.y4m', '.mp4'))
        command = f'ffmpeg -i {src_video} -c:v libx264 -crf {crf} {dst_video} -y'
        print(command)
        os.system(command)


def y4m_to_h265(src_path, dst_path, crf=22):
    src_videos = sorted(glob.glob(osp.join(src_path, '*.y4m')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.y4m', '.mp4'))
        command = f'ffmpeg -i {src_video} -c:v libx265 -crf {crf} {dst_video} -y'
        print(command)
        os.system(command)


def mp4_to_y4m(src_path, dst_path):
    src_videos = sorted(glob.glob(osp.join(src_path, '*.mp4')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.mp4', '.y4m'))
        command = f'ffmpeg -y -i {src_video} {dst_video}'
        print(command)
        os.system(command)


def avi_to_y4m(src_path, dst_path):
    src_videos = sorted(glob.glob(osp.join(src_path, '*.avi')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.avi', '.y4m'))
        command = f'ffmpeg -y -i {src_video} -pix_fmt yuv422p {dst_video}'
        print(command)
        os.system(command)


def h264():
    inp_video = 'inp_video.y4m'
    out_video = 'out_video.mp4'
    w, h = 1920, 1080
    crf_list = [18, 22, 26, 30, 34, 38, 42]
    for crf in crf_list:
        command = f'ffmpeg -y -i {inp_video} -vf scale={w}x{h} :flags=lanczos -c:v libx264 ' \
                  f'-profile:v high -threads 4 -preset veryslow -crf {crf} -refs 5 -g 150 ' \
                  f'-tune ssim -x264opts ssim=1 -keyint_min 150 -sc_threshold 0 -f mp4 ' \
                  f'{out_video}'
        print(command)
        os.system(command)


if __name__ == '__main__':
    src_path = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-original/360P-MP4'
    dst_path = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-original/360P-Y4M'
    # src_path = '/home/xiyang/data0/datasets/ReCp/VQEG/videos-original/VQEG-540P-MP4'
    # dst_path = '/home/xiyang/data0/datasets/ReCp/VQEG/videos-original/VQEG-540P-Y4M'
    mkdir(dst_path)
    # yuv_to_y4m(src_path, dst_path)
    # y4m_to_h264(src_path, dst_path)
    # y4m_to_h265(src_path, dst_path)
    mp4_to_y4m(src_path, dst_path)
    # avi_to_y4m(src_path, dst_path)