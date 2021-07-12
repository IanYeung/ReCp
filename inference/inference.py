import argparse
import os
import cv2
import glob
import time
import json
import numpy as np
import subprocess
import os.path as path

import logger
import ffmpeg

from scenedetect.detectors import ContentDetector
from scripts.scene_detection.scene_detect import MySceneManager

from enhancer import SingleFrameEnhancer, MultiFrameEnhancer
from basicsr.data.data_util import generate_frame_indices, generate_frame_indices_with_scene


def shot_det(frame_buf):
    a = [0]
    scene_manager = MySceneManager(input_mode='images')
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(frame_buf, step=1)
    res = scene_manager._get_cutting_list()
    a = a + res
    a.append(len(frame_buf))
    return a


def get_video_info(video_path):
    video_params = None
    audio_params = None

    command = 'ffprobe -v quiet -print_format json -show_format -show_streams {}'.format(video_path)
    status, video_info = subprocess.getstatusoutput(command)

    if status != 0:
        return [video_params, audio_params]

    try:
        video_info = json.loads(video_info)
    except:
        return [video_params, audio_params]

    for s in video_info['streams']:
        if s['codec_type'] == 'video':
            video_params = s
        elif s['codec_type'] == 'audio':
            audio_params = s
        else:
            pass

    return [video_params, audio_params]


def video_inference_sf_rgb(opt):
    input_video_list = sorted(glob.glob(opt.input_video))
    for input_video in input_video_list:
        cap = cv2.VideoCapture(input_video)
        assert cap.isOpened(), \
            '[{}] is a illegal input!'.format(input_video)

        # get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bitrate = opt.bitrate
        fps = '%.02f' % fps
        video_name = path.basename(input_video).split('.')[0]
        os.makedirs(opt.save_path, exist_ok=True)
        save_file = path.join(opt.save_path, '{}_srx2.mp4'.format(video_name))
        temp_file = path.join(opt.save_path, '{}_srx2_tmp.mp4'.format(video_name))

        ## get video codec info by ffprobe
        _, audio_params = get_video_info(input_video)

        # use python-ffmpeg
        reader = (ffmpeg
                    .input(input_video)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run_async(pipe_stdout=True)
        )
        writer = (ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width*2, height*2), r=fps)
                    .output(temp_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
        )

        # init model

        enhancer = SingleFrameEnhancer(model_opt_dict[opt.model_arch], opt.load_path, device=opt.device)

        k = 0
        t_sr = 0
        t1 = time.time()
        while True:
            k += 1
            in_bytes = reader.stdout.read(width * height * 3)
            if not in_bytes:
                print('Finish reading video')
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

            # inference on a frame
            t_start = time.time()
            out = enhancer.forward_single(frame)
            t_end = time.time()
            t_sr += t_end - t_start

            # write output to target video
            writer.stdin.write(out.tobytes())

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()
        cap.release()

        t2 = time.time()

        # concat audio and video
        if audio_params is not None:
            log.logger.info('Concat video and audio')
            input_ffmpeg = ffmpeg.input(input_video)
            audio = input_ffmpeg['a']
            output_ffmpeg = ffmpeg.input(temp_file)
            video = output_ffmpeg['v']

            if 'bit_rate' in audio_params.keys():
                a_bitrate = audio_params['bit_rate']
            else:
                a_bitrate = '128k'

            # rewrite video with audio
            mov = (ffmpeg
                    .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac')
                    )
            mov.overwrite_output().run()
            os.system('rm -f {}'.format(temp_file))

        else:
            os.system('mv {} {}'.format(temp_file, save_file))

        t3 = time.time()
        log.logger.info('============= Elapsed time =============')
        log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
        log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2-t1)/k*1000)))
        log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr/k*1000)))
        log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3-t2))
        log.logger.info('=========================================\n')


def video_inference_sf_yuv420(opt):
    input_video_list = sorted(glob.glob(opt.input_video))
    for input_video in input_video_list:
        cap = cv2.VideoCapture(input_video)
        assert cap.isOpened(), \
            '[{}] is a illegal input!'.format(input_video)

        # get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bitrate = opt.bitrate
        fps = '%.02f' % fps
        video_name = path.basename(input_video).split('.')[0]
        os.makedirs(opt.save_path, exist_ok=True)
        save_file = path.join(opt.save_path, '{}_x2.y4m'.format(video_name))
        temp_file = path.join(opt.save_path, '{}_x2_tmp.y4m'.format(video_name))

        ## get video codec info by ffprobe
        _, audio_params = get_video_info(input_video)

        # use python-ffmpeg
        reader = (ffmpeg
                    .input(input_video)
                    .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                    .run_async(pipe_stdout=True))
        writer = (ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w * 2, h * 2), r=fps)
                    .output(temp_file)
                    .overwrite_output()
                    .run_async(pipe_stdin=True))

        # init model

        enhancer = SingleFrameEnhancer(model_opt_dict[opt.model_arch], opt.load_path, device=opt.device)

        k = 0
        t_sr = 0
        t1 = time.time()
        while True:
            k += 1
            in_bytes_Y = reader.stdout.read(w * h)
            in_bytes_U = reader.stdout.read(w // 2 * h // 2)
            in_bytes_V = reader.stdout.read(w // 2 * h // 2)
            if not in_bytes_Y:
                print('Finish reading video')
                break
            Y = (np.frombuffer(in_bytes_Y, np.uint8).reshape([h, w]))
            U = (np.frombuffer(in_bytes_U, np.uint8).reshape([h // 2, w // 2]))
            V = (np.frombuffer(in_bytes_V, np.uint8).reshape([h // 2, w // 2]))

            # inference on a frame
            t_start = time.time()
            Y = enhancer.forward_single(Y, format='y')
            U = cv2.resize(U, (w, h), interpolation=cv2.INTER_CUBIC)
            V = cv2.resize(V, (w, h), interpolation=cv2.INTER_CUBIC)
            t_end = time.time()
            t_sr += t_end - t_start

            # write output to target video
            writer.stdin.write(Y.tobytes())
            writer.stdin.write(U.tobytes())
            writer.stdin.write(V.tobytes())

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()
        cap.release()

        t2 = time.time()

        # concat audio and video
        if audio_params is not None:
            log.logger.info('Concat video and audio')
            input_ffmpeg = ffmpeg.input(input_video)
            audio = input_ffmpeg['a']
            output_ffmpeg = ffmpeg.input(temp_file)
            video = output_ffmpeg['v']

            if 'bit_rate' in audio_params.keys():
                a_bitrate = audio_params['bit_rate']
            else:
                a_bitrate = '128k'

            # rewrite video with audio
            mov = (ffmpeg
                   .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac')
                   )
            mov.overwrite_output().run()
            os.system('rm -f {}'.format(temp_file))

        else:
            os.system('mv {} {}'.format(temp_file, save_file))

        t3 = time.time()
        log.logger.info('============= Elapsed time =============')
        log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
        log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2 - t1) / k * 1000)))
        log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr / k * 1000)))
        log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3 - t2))
        log.logger.info('=========================================\n')


def video_inference_mf_rgb(opt):
    input_video_list = sorted(glob.glob(opt.input_video))
    for input_video in input_video_list:
        cap = cv2.VideoCapture(input_video)
        assert cap.isOpened(), \
            '[{}] is a illegal input!'.format(input_video)

        # get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bitrate = opt.bitrate
        fps = '%.02f' % fps
        video_name = path.basename(input_video).split('.')[0]
        os.makedirs(opt.save_path, exist_ok=True)
        save_file = path.join(opt.save_path, '{}.mp4'.format(video_name))
        temp_file = path.join(opt.save_path, '{}_tmp.mp4'.format(video_name))

        cap.release()

        ## get video codec info by ffprobe
        _, audio_params = get_video_info(input_video)

        # use python-ffmpeg
        reader = (ffmpeg
                  .input(input_video)
                  .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                  .run_async(pipe_stdout=True))
        writer = (ffmpeg
                  .input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w, h), r=fps)
                  .output(temp_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                  .overwrite_output()
                  .run_async(pipe_stdin=True))

        # init model
        enhancer = MultiFrameEnhancer(model_opt_dict[opt.model_arch], opt.load_path, nframes=opt.nframes, device=opt.device)

        k = 0
        frame_buf = []
        buf_count = 0

        t_sr = 0
        t1 = time.time()
        while True:
            k += 1
            in_bytes = reader.stdout.read(width * height * 3)
            if not in_bytes:
                print('Finish reading video')
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            frame_buf.append(frame)
            enhancer.sequence_input_pool(frame)

            if len(frame_buf) < opt.frame_buf_len:
                continue
            else:
                ## shot det
                scene_list = shot_det(frame_buf)

                # inference on a seqnence
                for frame_idx in range(opt.frame_buf_len):
                    t_start = time.time()
                    select_idx = generate_frame_indices_with_scene(frame_idx, opt.frame_buf_len,
                                                                   opt.nframes, scene_list, padding='replicate')
                    out = enhancer.forward_sequence(select_idx)
                    t_end = time.time()
                    t_sr += t_end - t_start

                    # write output to target video
                    writer.stdin.write(out.tobytes())

                # clear buf
                buf_count += 1
                frame_buf.clear()
                enhancer.clear_input_cache()

        if len(frame_buf) > 0:
            ## shot det
            scene_list = shot_det(frame_buf)

            # inference on a seqnence
            for frame_idx in range(len(frame_buf)):
                t_start = time.time()
                select_idx = generate_frame_indices_with_scene(frame_idx, len(frame_buf), opt.nframes,
                                                               scene_list, padding='replicate')
                out = enhancer.forward_sequence(select_idx)
                t_end = time.time()
                t_sr += t_end - t_start

                # write output to target video
                writer.stdin.write(out.tobytes())

            # clear buf
            buf_count += 1
            frame_buf.clear()
            enhancer.clear_input_cache()

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()

        t2 = time.time()

        # concat audio and video
        if audio_params is not None:
            log.logger.info('Concat video and audio')
            input_ffmpeg = ffmpeg.input(input_video)
            audio = input_ffmpeg['a']
            output_ffmpeg = ffmpeg.input(temp_file)
            video = output_ffmpeg['v']

            if 'bit_rate' in audio_params.keys():
                a_bitrate = audio_params['bit_rate']
            else:
                a_bitrate = '128k'

            # rewrite video with audio
            mov = (ffmpeg
                   .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac'))
            mov.overwrite_output().run()
            os.system('rm -f {}'.format(temp_file))

        else:
            os.system('mv {} {}'.format(temp_file, save_file))

        t3 = time.time()
        log.logger.info('============= Elapsed time =============')
        log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
        log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2 - t1) / k * 1000)))
        log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr / k * 1000)))
        log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3 - t2))
        log.logger.info('=========================================\n')


def video_inference_mf_yuv420(opt):
    input_video_list = sorted(glob.glob(opt.input_video))
    for input_video in input_video_list:
        cap = cv2.VideoCapture(input_video)
        assert cap.isOpened(), \
            '[{}] is a illegal input!'.format(input_video)

        # get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bitrate = opt.bitrate
        fps = '%.02f' % fps
        video_name = path.basename(input_video).split('.')[0]
        os.makedirs(opt.save_path, exist_ok=True)
        save_file = path.join(opt.save_path, '{}.mp4'.format(video_name))
        temp_file = path.join(opt.save_path, '{}_tmp.mp4'.format(video_name))

        cap.release()

        ## get video codec info by ffprobe
        _, audio_params = get_video_info(input_video)

        # use python-ffmpeg
        reader = (ffmpeg
                    .input(input_video)
                    .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                    .run_async(pipe_stdout=True))
        writer = (ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w, h), r=fps)
                    .output(temp_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                    .overwrite_output()
                    .run_async(pipe_stdin=True))

        # init model
        enhancer = MultiFrameEnhancer(model_opt_dict[opt.model_arch], opt.load_path, nframes=opt.nframes, device=opt.device)

        k = 0
        frame_buf = []
        buf_count = 0

        t_sr = 0
        t1 = time.time()
        while True:
            k += 1
            in_bytes_Y = reader.stdout.read(w * h)
            in_bytes_U = reader.stdout.read(w // 2 * h // 2)
            in_bytes_V = reader.stdout.read(w // 2 * h // 2)
            if not in_bytes_Y:
                print('Finish reading video')
                break
            Y = (np.frombuffer(in_bytes_Y, np.uint8).reshape([h, w]))
            U = (np.frombuffer(in_bytes_U, np.uint8).reshape([h // 2, w // 2]))
            V = (np.frombuffer(in_bytes_V, np.uint8).reshape([h // 2, w // 2]))
            YUV = np.zeros((h, w, 3), dtype=np.uint8)
            # Y channel
            YUV[:, :, 0] = Y
            # U channel
            YUV[0::2, 0::2, 1] = U
            YUV[0::2, 1::2, 1] = U
            YUV[1::2, 0::2, 1] = U
            YUV[1::2, 1::2, 1] = U
            # V channel
            YUV[0::2, 0::2, 2] = V
            YUV[0::2, 1::2, 2] = V
            YUV[1::2, 0::2, 2] = V
            YUV[1::2, 1::2, 2] = V
            # frame = YUV / 255.
            frame = YUV
            frame_buf.append(frame)
            enhancer.sequence_input_pool(frame)

            if len(frame_buf) < opt.frame_buf_len:
                continue
            else:
                ## shot det
                scene_list = shot_det(frame_buf)

                # inference on a seqnence
                for frame_idx in range(opt.frame_buf_len):
                    t_start = time.time()
                    select_idx = generate_frame_indices_with_scene(frame_idx, opt.frame_buf_len,
                                                                   opt.nframes, scene_list, padding='replicate')
                    out = enhancer.forward_sequence(select_idx)
                    t_end = time.time()
                    t_sr += t_end - t_start

                    # write output to target video
                    out = out.astype(np.float32)
                    Y = out[:, :, 0]
                    U = (out[0::2, 0::2, 1] + out[0::2, 1::2, 1] + out[1::2, 0::2, 1] + out[1::2, 1::2, 1]) / 4
                    V = (out[0::2, 0::2, 2] + out[0::2, 1::2, 2] + out[1::2, 0::2, 2] + out[1::2, 1::2, 2]) / 4
                    Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
                    writer.stdin.write(Y.tobytes())
                    writer.stdin.write(U.tobytes())
                    writer.stdin.write(V.tobytes())

                # clear buf
                buf_count += 1
                frame_buf.clear()
                enhancer.clear_input_cache()

        if len(frame_buf) > 0:
            ## shot det
            scene_list = shot_det(frame_buf)

            # inference on a seqnence
            for frame_idx in range(len(frame_buf)):
                t_start = time.time()
                select_idx = generate_frame_indices_with_scene(frame_idx, len(frame_buf), opt.nframes,
                                                               scene_list, padding='replicate')
                out = enhancer.forward_sequence(select_idx)
                t_end = time.time()
                t_sr += t_end - t_start

                # write output to target video
                out = out.astype(np.float32)
                Y = out[:, :, 0]
                U = (out[0::2, 0::2, 1] + out[0::2, 1::2, 1] + out[1::2, 0::2, 1] + out[1::2, 1::2, 1]) / 4
                V = (out[0::2, 0::2, 2] + out[0::2, 1::2, 2] + out[1::2, 0::2, 2] + out[1::2, 1::2, 2]) / 4
                Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
                writer.stdin.write(Y.tobytes())
                writer.stdin.write(U.tobytes())
                writer.stdin.write(V.tobytes())

            # clear buf
            buf_count += 1
            frame_buf.clear()
            enhancer.clear_input_cache()

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()

        t2 = time.time()

        # concat audio and video
        if audio_params is not None:
            log.logger.info('Concat video and audio')
            input_ffmpeg = ffmpeg.input(input_video)
            audio = input_ffmpeg['a']
            output_ffmpeg = ffmpeg.input(temp_file)
            video = output_ffmpeg['v']

            if 'bit_rate' in audio_params.keys():
                a_bitrate = audio_params['bit_rate']
            else:
                a_bitrate = '128k'

            # rewrite video with audio
            mov = (ffmpeg
                    .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac'))
            mov.overwrite_output().run()
            os.system('rm -f {}'.format(temp_file))

        else:
            os.system('mv {} {}'.format(temp_file, save_file))

        t3 = time.time()
        log.logger.info('============= Elapsed time =============')
        log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
        log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2-t1)/k*1000)))
        log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr/k*1000)))
        log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3-t2))
        log.logger.info('=========================================\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Restoration')
    parser.add_argument('--input_video',
                        default='/home/xiyang/Datasets/test/test_videos/lq_videos_tmp/*')
    parser.add_argument('--save_path',
                        default='/home/xiyang/Datasets/test/test_videos/hq_videos_tmp')
    parser.add_argument('--model_arch', type=str, default='MSRResNet_Y')
    parser.add_argument('--load_path', default='../experiments/pretrained_models/MSRResNet_x2_Y_JointTrain.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mode', type=str, default='yuv')
    parser.add_argument('--bitrate', type=str, default='10M')
    parser.add_argument('--nframes', type=int, default=1)
    parser.add_argument('--frame_buf_len', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='../log/inference.log')
    opt = parser.parse_args()

    log = logger.Logger(filename=opt.log_path, level='debug')
    log.logger.info(opt)

    model_opt_dict = {
        'MSRResNet_Y': {'type': 'MSRResNet', 'num_feat': 64, 'num_block': 16, 'num_in_ch': 1, 'num_out_ch': 1,
                        'upscale': 2},
        'MSRResNet': {'type': 'MSRResNet', 'num_feat': 64, 'num_block': 16, 'num_in_ch': 3, 'num_out_ch': 3,
                      'upscale': 2}
    }

    if opt.nframes == 1:
        print('process on single-frames mode')
        if opt.mode == 'rgb':
            video_inference_sf_rgb(opt)
        else:
            video_inference_sf_yuv420(opt)
    else:
        print('process on multi-frames mode')
        if opt.mode == 'rgb':
            video_inference_mf_rgb(opt)
        else:
            video_inference_mf_yuv420(opt)
