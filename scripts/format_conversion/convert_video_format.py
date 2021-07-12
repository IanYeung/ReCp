import os
import os.path as osp
import sys
import glob


def yuv_to_y4m():
    src_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'
    dst_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'

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


def y4m_to_h264(crf=22):
    src_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'
    dst_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'

    src_videos = sorted(glob.glob(osp.join(src_path, '*.y4m')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.y4m', '.mp4'))
        command = f'ffmpeg -i {src_video} -c:v libx264 -crf {crf} {dst_video} -y'
        print(command)
        os.system(command)


def y4m_to_h265(crf=22):
    src_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'
    dst_path = '/home/xiyang/data0/datasets/Video-Compression-Datasets/MCL-JVC/720P'

    src_videos = sorted(glob.glob(osp.join(src_path, '*.y4m')))
    for src_video in src_videos:
        dst_video = osp.join(dst_path, osp.basename(src_video).replace('.y4m', '.mp4'))
        command = f'ffmpeg -i {src_video} -c:v libx265 -crf {crf} {dst_video} -y'
        print(command)
        os.system(command)


if __name__ == '__main__':
    yuv_to_y4m()
    # y4m_to_h264()
    # y4m_to_h265()
