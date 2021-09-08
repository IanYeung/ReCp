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


VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mkv', '.MKV', '.mpg', '.MPG', '.mxf', '.MXF', '.y4m', '.Y4M']


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def encode_frames_with_ffmpeg(src_path, dst_path, crf, fps=25, start_number=1, vframes=1000):
    command = 'ffmpeg -r {} -f image2 -start_number {} -i {} -vframes {} -vcodec libx265 ' \
              '-vf fps={} -crf {} -pix_fmt yuv420p -an {} -y &>/dev/null'\
        .format(fps, start_number, src_path, vframes, fps, crf, dst_path)
    print('doing... ' + command)
    os.system(command)


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


def compress_h264(inp_video, out_video, params):
    command = 'ffmpeg -y -i {} -vf scale={}x{}:flags=lanczos -c:v libx264 ' \
              '-profile:v high -threads 4 -preset veryslow -crf {} -refs 5 -g 150 ' \
              '-tune ssim -x264opts ssim=1 -keyint_min 150 -sc_threshold 0 -f mp4 ' \
              '{}'.format(inp_video, params['W'], params['H'], params['crf'], out_video)
    print(command)
    os.system(command)


def lossless_h264(inp_video, out_video):
    command = 'ffmpeg -y -i {} -c:v libx264 -preset veryslow -crf 0 ' \
              '{}'.format(inp_video, out_video)
    print(command)
    os.system(command)


def process(inp_video, out_video, params):

    # open video
    if is_video_file(inp_video):
        print('---------------' * 4)
        print('Opening video [{}]'.format(inp_video))
        cap = cv2.VideoCapture(inp_video)
        assert cap.isOpened(), '[{}] is a illegal input!'.format(inp_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        raise NotImplementedError('Please input a video file!')

    # set target attribute
    mode = params['mode']
    fps = '%.02f' % fps

    ow = iw // params['down_scale']
    oh = ih // params['down_scale']

    # use ffmpeg-python
    reader = (ffmpeg
              .input(inp_video)
              .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
              .run_async(pipe_stdout=True))

    if mode == 'bitrate':
        bitrate = '%dk' % params['bitrate']
        writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='yuv420p',
                               s='{}x{}'.format(ow, oh), r=fps)
                  .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                          s='{}x{}'.format(ow, oh), video_bitrate=bitrate, r=fps)
                  .overwrite_output()
                  .run_async(pipe_stdin=True)
                  )
    elif mode == 'crf':
        crf = params['crf']
        writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='yuv420p',
                               s='{}x{}'.format(ow, oh), r=fps)
                  .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                          s='{}x{}'.format(ow, oh), crf=crf, r=fps)
                  .overwrite_output()
                  .run_async(pipe_stdin=True)
                  )
    elif mode == 'qp':
        qp = params['qp']
        writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='yuv420p',
                               s='{}x{}'.format(ow, oh), r=fps)
                  .output(out_video, vcodec='libx264', pix_fmt='yuv420p',
                          s='{}x{}'.format(ow, oh), qp=qp, r=fps)
                  .overwrite_output()
                  .run_async(pipe_stdin=True)
                  )
    else:
        raise ValueError('Mode {} has not been implemented.'.format(mode))

    # processing
    t1 = time.time()

    while True:
        # read a video frame
        in_bytes_Y = reader.stdout.read(iw * ih)
        in_bytes_U = reader.stdout.read(iw // params['down_scale'] * ih // params['down_scale'])
        in_bytes_V = reader.stdout.read(iw // params['down_scale'] * ih // params['down_scale'])
        if not in_bytes_Y:
            print('Finished reading video.')
            break
        Y = (np.frombuffer(in_bytes_Y, np.uint8).reshape([ih, iw]))
        U = (np.frombuffer(in_bytes_U, np.uint8).reshape([ih // params['down_scale'], iw // params['down_scale']]))
        V = (np.frombuffer(in_bytes_V, np.uint8).reshape([ih // params['down_scale'], iw // params['down_scale']]))

        Y, U, V = Y / 255., U / 255., V / 255.
        # degradation on a frame
        Y = resizer.imresize(Y, scale_factor=1.0 / params['down_scale'], output_shape=None, kernel='cubic',
                             antialiasing=True, kernel_shift_flag=False)
        U = resizer.imresize(U, scale_factor=1.0 / params['down_scale'], output_shape=None, kernel='cubic',
                             antialiasing=True, kernel_shift_flag=False)
        V = resizer.imresize(V, scale_factor=1.0 / params['down_scale'], output_shape=None, kernel='cubic',
                             antialiasing=True, kernel_shift_flag=False)

        Y, U, V = (np.clip(Y, 0, 1) * 255.0).round(), (np.clip(U, 0, 1) * 255.0).round(), (np.clip(V, 0, 1) * 255.0).round()
        # write output to target video
        Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
        writer.stdin.write(Y.tobytes())
        writer.stdin.write(U.tobytes())
        writer.stdin.write(V.tobytes())

    reader.stdout.close()
    writer.stdin.close()
    writer.wait()
    t2 = time.time()

    print('Finished saving video [{}], cost timg: {:.2f}s'.format(out_video, (t2 - t1)))


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
                # print('{}X downsample and mpeg compression with ffmpeg'.format(scale))
                pass
            elif down_type == 'cv_cubic':
                # print('{}X down-sample with cv2 bicubic'.format(scale))
                img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), cv2.INTER_CUBIC)
            elif down_type == 'mat_cubic':
                # print('{}X down-sample with matlab bicubic'.format(scale))
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


def degrade_vimeo90k(root='/data2/yangxi/datasets/Vimeo90k', name='vimeo_septuplet_BIx2_h264_crf23_img_SRx2'):
    vid = VideoDegrader()

    # fixed degradation (Vimeo90K)
    scale = 1
    mode = 'crf'
    for quality in [18, 22, 26, 30]:
        params = {'is_blur': False, 'blur_type': 'gaussian', 'blur_size': 9, 'blur_sigma': 5,
                  'is_noise': False, 'noise_sigma': 5.0,
                  'down_type': 'mat_cubic', 'down_scale': scale,
                  'mode': mode, 'bitrate': quality, 'crf': quality, 'qp': quality}

        read_root_img = os.path.join(root, '{}'.format(name), 'sequences')
        save_root_mp4 = os.path.join(root, '{}_{}{}_mp4'.format(name, mode, quality), 'sequences')
        save_root_img = os.path.join(root, '{}_{}{}_img'.format(name, mode, quality), 'sequences')

        folder_list = [osp.basename(folder) for folder in sorted(glob.glob(osp.join(read_root_img, '*')))]
        # folder_list = [folder for folder in folder_list if folder == '00028']

        for folder in folder_list:
            # encode
            seq_path_list = sorted(glob.glob(osp.join(read_root_img, folder, '*')))
            mkdir(osp.join(save_root_mp4, folder))
            for seq_path in seq_path_list:
                seq_name = osp.basename(seq_path)
                vid.process(seq_path, osp.join(save_root_mp4, folder, '{}.mp4'.format(seq_name)), params=params)
                print(vid.get_params())
            # # decode
            # seq_path_list = sorted(glob.glob(osp.join(save_root_mp4, folder, '*')))
            # mkdir(osp.join(save_root_img, folder))
            # for seq_path in seq_path_list:
            #     seq_name = osp.basename(seq_path).split('.')[0]
            #     print('Processing: {}'.format(seq_name))
            #     frm_path = osp.join(save_root_img, folder, seq_name)
            #     mkdir(frm_path)
            #     decode_frames_with_ffmpeg(seq_path, frm_path)


def degrade_mcl_jvc():
    # video folder (y4m/mp4)
    params = {'down_type': 'mat_cubic', 'down_scale': 2, 'mode': 'crf', 'crf': 23}
    video_src_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-original/720P-Y4M'
    video_dst_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-original/360P-MP4'
    # video_src_root = '/home/xiyang/data0/datasets/ReCp/VQEG/videos-original/VQEG-1080P-Y4M'
    # video_dst_root = '/home/xiyang/data0/datasets/ReCp/VQEG/videos-original/VQEG-540P-MP4'
    mkdir(video_dst_root)
    inp_video_list = sorted(glob.glob(os.path.join(video_src_root, '*')))
    for inp_video_path in inp_video_list:
        out_video_path = osp.join(video_dst_root, os.path.basename(inp_video_path).replace('.y4m', '.mp4'))
        process(inp_video_path, out_video_path, params=params)


def compress_mcl_jvc(root='/home/xiyang/data0/datasets/ReCp/MCL-JVC'):
    # model = 'MSRResNet_x2_Vimeo90k_250k_Y'
    # model = 'MSRResNet_DoubleFrameCompressor_x2_Vimeo90k_250k_Y_ratio_1.0_1.0_mix'
    # model = 'MSRResNet_DoubleFrameCompressor_x2_Vimeo90k_250k_Y_sr1.0_cp1.0_rate0.01_mix'
    # model = 'PrecodingResNet_DoubleFrameCompressor_Vimeo90k_250k_Y_Lf1.0_Lr0.001_mix'
    model = 'PrecodingResNet_DoubleFrameCompressor_Vimeo90k_250k_Y_Lf1.0_Lr0.005_mix'

    crf_list = [18, 22, 26, 30, 34, 38, 42]
    for crf in crf_list:
        params = {'W': 1280, 'H': 720, 'crf': crf}
        # video_src_root = os.path.join(root, 'videos-original/720P-Y4M-{}'.format(model))
        # video_dst_root = os.path.join(root, 'videos-compress/{}/CRF{}'.format(model, params['crf']))
        video_src_root = os.path.join(root, 'videos-original/720P-Y4M')
        video_dst_root = os.path.join(root, 'videos-compress/{}/CRF{}'.format('Baseline', params['crf']))
        mkdir(video_dst_root)
        inp_video_list = sorted(glob.glob(os.path.join(video_src_root, '*')))
        for inp_video_path in inp_video_list:
            out_video_path = osp.join(video_dst_root, os.path.basename(inp_video_path).replace('.y4m', '.mp4'))
            compress_h264(inp_video_path, out_video_path, params=params)


if __name__ == "__main__":
    # root = '/data2/yangxi/datasets/Vimeo90k'
    # name = 'vimeo_septuplet_BIx2_h264_crf23_img_SRx2'
    # degrade_vimeo90k(root, name)

    compress_mcl_jvc()
