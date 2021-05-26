import torch
import os
import cv2
import sys
import glob
import time
import shutil
import numpy as np
import subprocess
import ffmpeg
import random
import resizer
from os import path as osp
from scipy.io import loadmat


VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mkv', '.MKV', '.mpg', '.MPG', '.mxf', '.MXF']


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def encode_frames_with_ffmpeg(src_path, dst_path, crf, fps=25, start_number=1, vframes=1000):
#     command = 'ffmpeg -r {} -f image2 -start_number {} -i {} -vframes {} -vcodec libx265 ' \
#               '-vf fps={} -crf {} -pix_fmt yuv420p -an {} -y &>/dev/null'\
#         .format(fps, start_number, src_path, vframes, fps, crf, dst_path)
#     print('doing... ' + command)
#     os.system(command)


def encode_frames_with_ffmpeg(src_path, dst_path, mode='crf'):
    if mode == 'crf':
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -pix_fmt yuv420p -crf 23 {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    elif mode == 'cqp':
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -pix_fmt yuv420p -crf 23 {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    elif mode == 'abr':
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -pix_fmt yuv420p -b:v 400k {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    elif mode == 'vbr':
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -pix_fmt yuv420p -b:v 400k -pass 1 -f null /dev/null'.format(src_path)
        print(command)
        os.system(command)
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -pix_fmt yuv420p -b:v 400k -pass 2 {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    elif mode == 'cbr':
        command = 'ffmpeg -r 25 -f image2 -i {} -c:v libx264 -x264-params "nal-hrd=cbr:force-cfr=1" ' \
                  '-b:v 1M -minrate 1M -maxrate 1M -bufsize 2M {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    else:
        raise ValueError()


def decode_frames_with_ffmpeg(video_path, image_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'cannot open video {}'.format(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True)
    )

    k = 0
    while True:
        k += 1
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        cv2.imwrite(osp.join(image_path, 'im{}.png'.format(k)), frame)

    process.wait()
    print('total {} frames'.format(k - 1))


class VideoDegrader(object):
    """
    End-to-End video degradation.
    """

    def __init__(self):
        super(VideoDegrader, self).__init__()
        self.deg_params = {}
        self._init_deg_params()
        print('VideoDegrader init done, [deg_params] init done')

    def _init_deg_params(self):
        self.deg_params = {
            # blur
            'is_blur': True,  # do blur degradation
            'blur_type': 'gaussian',  # blur type, now support ['gaussian', 'motion'] blur
            'blur_size': 1,  # gaussian blur size
            'blur_sigma': 1.0,  # gaussian blur sigma
            'blur_kernel': 1,  # motion blur kernel [1, 32]

            # downsample
            'down_scale': 1,  # scale factor for downsample, 1 is not downscale
            'down_type': 'ffmpeg',  # down sample type, choose from ['ffmpeg', 'mat_cubic', 'cv_cubic']

            # noise
            'is_noise': True,  # add noise
            'noise_type': 'gaussian',  # noise type, only support gaussian now
            'noise_sigma': 1.0,  # gaussian noise sigma

            # video encoder
            'mode': 'bitrate',
            'bitrate': 600,  # bitrate (K) for encoding LR video
            'crf': 23,       # crf for encoding LR video
            'qp': 23,        # qp for encoding LR video
        }

    def update_deg_params(self, params=None):
        if params is not None:
            print('update degradation parameters via input')
            self.deg_params.update(params)
        else:
            print('update degradation parameters randomly')
            # blur params
            self.deg_params['is_blur'] = True if random.random() < 0.7 else False
            self.deg_params['blur_type'] = 'gaussian' if random.random() < 0.5 else 'motion'
            self.deg_params['blur_size'] = random.randint(1, 6) * 2 + 1  # 3, 5, 7, 9, 11, 13
            self.deg_params['blur_sigma'] = float(random.randint(1, 7))
            self.deg_params['blur_kernel'] = random.randint(1, 32)

            # downsample params
            self.deg_params['down_scale'] = 2
            # M = random.random()
            # if M < 0.3:
            #     self.deg_params['down_type'] = 'cv_cubic'
            # elif M < 0.6:
            #     self.deg_params['down_type'] = 'mat_cubic'
            # else:
            #     self.deg_params['down_type'] = 'ffmpeg'
            self.deg_params['down_type'] = 'cv_cubic'

            # noise params
            self.deg_params['is_noise'] = True if random.random() < 0.3 else False
            self.deg_params['noise_type'] = 'gaussian'
            self.deg_params['noise_sigma'] = float(random.randint(1, 11))

            # video compression
            self.deg_params['mode'] = 'bitrate'
            self.deg_params['bitrate'] = 600 + random.randint(0, 50) * 20  # [600k, 1600k]
            self.deg_params['crf'] = 23
            self.deg_params['qp'] = 23

    def deg_image(self, img):
        """
        input : img(np.array(uint8))
        return: degraded img(np.array(uint8))
        degradation pipeline:
            blur -> downsample (cubic) -> noise -> bitrate (ffmpeg)
            blur -> noise -> downsample + bitrate (ffmpeg)
        """
        # blur
        if self.deg_params['is_blur']:
            blur_type = self.deg_params['blur_type']
            if blur_type == 'gaussian':
                blur_size = self.deg_params['blur_size']
                blur_sigma = float(self.deg_params['blur_sigma']) / 10.0
                # print('Add gaussian blur with size [{}] and sigma [{}]'.format(blur_size, blur_sigma))
                img = cv2.GaussianBlur(img, (blur_size, blur_size), blur_sigma)
            elif blur_type == 'motion':
                kernel = self.deg_params['blur_kernel']
                # print('Add motion blur with kernel [{}]'.format(kernel))
                KName = './MotionBlurKernel/m_%02d.mat' % kernel
                k = loadmat(KName)['kernel']
                k = k.astype(np.float32)
                k /= np.sum(k)
                img = cv2.filter2D(img, -1, k)
            else:
                raise NotImplementedError('Blur type [{:s}] not recognized'.format(blur_type))
            img = np.clip(img, 0, 255)

        # downsample
        scale = self.deg_params['down_scale']
        if scale == 1:
            print('No need to downscale')
        else:
            down_type = self.deg_params['down_type']
            if down_type == 'ffmpeg':
                print('{}X downsample and mpeg compression with ffmpeg'.format(scale))
            elif down_type == 'cv_cubic':
                print('{}X down-sample with cv2 bicubic'.format(scale))
                img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), cv2.INTER_CUBIC)
            elif down_type == 'mat_cubic':
                print('{}X down-sample with matlab bicubic'.format(scale))
                img = resizer.imresize(img, scale_factor=1.0 / scale, output_shape=None, kernel='cubic',
                                       antialiasing=True, kernel_shift_flag=False)
            else:
                raise NotImplementedError('Downsample type [{:s}] not recognized'.format(down_type))

        # noise
        if self.deg_params['is_noise']:
            noise_type = self.deg_params['noise_type']
            if noise_type == 'gaussian':
                sigma = self.deg_params['noise_sigma']
                # print('Add gaussian noise with sigma {}'.format(sigma))
                img_tensor = torch.from_numpy(np.array(img)).float()
                noise = torch.randn(img_tensor.size()).mul_(sigma)
                noise_tensor = torch.clamp(noise + img_tensor, 0, 255)
                img = np.uint8(noise_tensor.numpy())
            else:
                raise NotImplementedError('Noise type [{:s}] not recognized'.format(noise_type))

        return img

    def process(self, in_video, out_video, params=None):
        """
        hq video in, lq video out
        decode hq video and degrade frame by frame, encode lq video via ffmpeg
        if in_video is a image folder, directly read the image as a sequence
        """
        self.update_deg_params(params)

        # open video
        if is_video_file(in_video):
            print('Input video is a file, use video processor.')
            print('open video [{}]'.format(in_video))
            cap = cv2.VideoCapture(in_video)
            assert cap.isOpened(), '[{}] is a illegal input!'.format(in_video)
            # get video info
            is_video = True
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            print('Input video is a folder, use img list processor.')
            is_video = False
            img_list = sorted(glob.glob(osp.join(in_video, '*.png')))
            height, width, c = cv2.imread(img_list[0]).shape
            assert c == 3, 'input image must have 3 channels!'
            fps = 25

        # set target attribute
        mode = self.deg_params['mode']
        bitrate = '%dk' % self.deg_params['bitrate']
        crf = self.deg_params['crf']
        qp = self.deg_params['qp']
        fps = '%.02f' % fps
        tw = width // self.deg_params['down_scale']
        th = height // self.deg_params['down_scale']

        # use ffmpeg-python
        if is_video:
            reader = (ffmpeg
                      .input(in_video)
                      .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                      .run_async(pipe_stdout=True))
        if self.deg_params['down_type'] == 'ffmpeg':
            writer_ih, writer_iw = height, width
        else:
            writer_ih, writer_iw = th, tw
        if mode == 'bitrate':
            writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                                   s='{}x{}'.format(writer_iw, writer_ih), r=fps)
                      .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                              s='{}x{}'.format(tw, th), video_bitrate=bitrate, r=fps)
                      .overwrite_output()
                      .run_async(pipe_stdin=True)
                      )
        elif mode == 'crf':
            writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                                   s='{}x{}'.format(writer_iw, writer_ih), r=fps)
                      .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                              s='{}x{}'.format(tw, th), crf=crf, r=fps)
                      .overwrite_output()
                      .run_async(pipe_stdin=True)
                      )
        elif mode == 'qp':
            writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                                   s='{}x{}'.format(writer_iw, writer_ih), r=fps)
                      .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                              s='{}x{}'.format(tw, th), qp=qp, r=fps)
                      .overwrite_output()
                      .run_async(pipe_stdin=True)
                      )
        else:
            raise ValueError('Mode {} has not been implemented.'.format(mode))

        # processing
        t1 = time.time()
        k = 0
        while True:
            if is_video:
                # read a video frame
                in_bytes = reader.stdout.read(width * height * 3)
                if not in_bytes:
                    print('Finish reading video')
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            else:
                if k >= len(img_list):
                    print('Finish reading sequence')
                    break
                frame = cv2.imread(img_list[k])
                k += 1

            # degradation on a frame
            deg_frame = self.deg_image(frame)
            # deg_frame = deg_frame[:, :, [2, 1, 0]]  # BGR2RGB

            # write output to target video
            writer.stdin.write(deg_frame.astype(np.uint8).tobytes())

        if is_video:
            reader.stdout.close()
        writer.stdin.close()
        writer.wait()
        t2 = time.time()
        print('save video [{}], cost timg: {:.2f}s'.format(out_video, (t2 - t1)))

    def get_params(self):
        return self.deg_params


if __name__ == "__main__":
    vid = VideoDegrader()

    # # randomly degradation (Alibaba-SR)
    # vid.process('tmp/01RedSea_1920x800_20M-001_1.mp4', 'tmp/out.mp4')
    # print(vid.get_params())

    # fixed degradation (Vimeo90K)
    scale = 1
    mode = 'crf'
    quality = 33
    params = {'is_blur': False, 'blur_type': 'gaussian', 'blur_size': 9, 'blur_sigma': 5,
              'is_noise': False, 'noise_sigma': 5.0,
              'down_type': 'mat_cubic', 'down_scale': scale,
              'mode': mode, 'bitrate': quality, 'crf': quality, 'qp': quality}

    # read_root_img = f'/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_enh/sequences'
    # save_root_mp4 = f'/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_com_{mode}{quality}_mp4/sequences'
    # save_root_img = f'/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet_BIx2_h264_crf23_com_{mode}{quality}_img/sequences'

    # exp_name = '001_MSRResNet_x2_f64b16_Vimeo90k_250k_B16G1_wandb'
    # exp_name = '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf23'
    exp_name = '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf28'
    # exp_name = '001_MSRResNet_BIC_x2_Vimeo90k_250k_crf23'
    # exp_name = '001_MSRResNet_BIC_x2_Vimeo90k_250k_crf28'

    read_root_img = f'/home/xiyang/Datasets/ReCp/results/{exp_name}/visualization/Vimeo90K'
    save_root_mp4 = f'/home/xiyang/Datasets/ReCp/results/{exp_name}/visualization/Vimeo90k_{mode}{quality}_video'
    save_root_img = f'/home/xiyang/Datasets/ReCp/results/{exp_name}/visualization/Vimeo90k_{mode}{quality}_frame'

    folder_list = [osp.basename(folder) for folder in sorted(glob.glob(osp.join(read_root_img, '*')))]
    for folder in folder_list:
        # encode
        seq_path_list = sorted(glob.glob(osp.join(read_root_img, folder, '*')))
        mkdir(osp.join(save_root_mp4, folder))
        for seq_path in seq_path_list:
            seq_name = osp.basename(seq_path)
            vid.process(seq_path, osp.join(save_root_mp4, folder, '{}.mp4'.format(seq_name)), params=params)
            print(vid.get_params())
        # decode
        seq_path_list = sorted(glob.glob(osp.join(save_root_mp4, folder, '*')))
        mkdir(osp.join(save_root_img, folder))
        for seq_path in seq_path_list:
            seq_name = osp.basename(seq_path).split('.')[0]
            print('Processing: {}'.format(seq_name))
            frm_path = osp.join(save_root_img, folder, seq_name)
            mkdir(frm_path)
            decode_frames_with_ffmpeg(seq_path, frm_path)

    # # fixed degradation (SPMCS)
    # scale = 2
    # qp = 32
    # params = {'is_blur': False, 'blur_type': 'gaussian', 'blur_size': 9, 'blur_sigma': 5,
    #           'is_noise': False, 'noise_sigma': 5.0,
    #           'down_type': 'mat_cubic', 'down_scale': scale,
    #           'mode': 'crf', 'bitrate': 200, 'crf': 23, 'qp': qp}
    #
    # root = '/home/xiyang/Datasets/SPMCS'
    #
    # seq_list = sorted(glob.glob(osp.join(root, '*')))
    # for seq in seq_list:
    #     # encode
    #     seq_name = osp.basename(seq)
    #     print('Processing: {}'.format(seq_name))
    #     read_root_img = osp.join(root, seq_name, 'GT')
    #     save_path_mp4 = osp.join(root, seq_name, '{}.mp4'.format(seq_name))
    #     vid.process(read_root_img, save_path_mp4, params=params)
    #     print(vid.get_params())
    #     # decode
    #     save_root_img = osp.join(root, seq_name, 'BIX{}_compressed_img'.format(scale))
    #     mkdir(save_root_img)
    #     decode_frames_with_ffmpeg(save_path_mp4, save_root_img)

