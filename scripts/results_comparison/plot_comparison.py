import os
import os.path as osp
import cv2
import copy
import numpy as np

from PIL import ImageDraw, ImageFont, Image


def draw_text(frame, text, position='bl-br', color=(255, 255, 255)):
    fontSizeR = 0.05
    w = frame.shape[1]
    h = frame.shape[0]
    fontFile = 'Alibaba-PuHuiTi-Regular.otf'
    ############################################
    font = ImageFont.truetype(fontFile, int(min(w, h) * fontSizeR))
    ############################################
    image = Image.fromarray(frame, 'RGB')
    draw = ImageDraw.Draw(image)
    draw.font = font
    ts = draw.textsize(text)
    ts = (ts[0], max(ts[1], draw.textsize('gh')[1]))
    b = ts[1] * 0.2
    shadowSize = max(1, font.size/20)
    poses = position.split('-')
    for pos in poses:
        if pos[0] == 't':
            y = h * 0.01
        elif pos[0] == 'b':
            y = h * (1-0.01) - ts[1]
        else:# m
            y = (h - ts[1]) / 2
        if pos[1] == 'l':
            x = w * 0.01
        elif pos[1] == 'r':
            x = w * (1-0.01) - ts[0]
        else:# m
            x = (w - ts[0]) / 2
        draw.rectangle([(x-b, y-b), (x+ts[0]+b, y+ts[1]+b)], fill=(36, 90, 241))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        draw.text((x+shadowSize, y), text, fill=color, font=font)
    frame = copy.deepcopy(np.ascontiguousarray(np.asarray(image).reshape(image.size[1], image.size[0], 3)))
    return frame


def drawLine(frame, x=-1, y=-1):
    lineWidthR = 0.002
    lineWidth = min(int(lineWidthR * frame.shape[0] * 2),int(lineWidthR * frame.shape[1]))
    if x >= 0:
        frame[:, x-lineWidth:x+lineWidth, :] = 255
    elif y >= 0:
        frame[y-lineWidth:y+lineWidth, :, :] = 255


def compare_vimeo90k_imgx6(src_root, dst_root, name_list, experiment_list, crf_list):
    assert len(experiment_list) == 5
    for name in name_list:
        name_1, name_2 = name.split('/')
        for crf in crf_list:
            h, w, c = 256, 448, 3
            out = np.zeros((2 * h, 3 * w, c), dtype=np.uint8)

            # ground truth
            path = osp.join(src_root, 'GT', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, 'GT', position='tl')
            out[0*h:1*h, 0*w:1*w, :] = img

            # baseline
            path = osp.join(src_root, experiment_list[0], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, 'Baseline', position='tl')
            out[1*h:2*h, 0*w:1*w, :] = img

            # crf 18
            path = osp.join(src_root, experiment_list[1], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, f'Joint Training (CRF {crf_list[0]})', position='tl')
            out[0*h:1*h, 1*w:2*w, :] = img

            # crf 23
            path = osp.join(src_root, experiment_list[2], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, f'Joint Training (CRF {crf_list[1]})', position='tl')
            out[1*h:2*h, 1*w:2*w, :] = img

            # crf 28
            path = osp.join(src_root, experiment_list[3], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, f'Joint Training (CRF {crf_list[2]})', position='tl')
            out[0*h:1*h, 2*w:3*w, :] = img

            # crf 33
            path = osp.join(src_root, experiment_list[4], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, f'Joint Training (CRF {crf_list[3]})', position='tl')
            out[1*h:2*h, 2*w:3*w, :] = img

            dst_path = osp.join(dst_root, f'{name_1}_{name_2}_crf{crf}.png')
            cv2.imwrite(dst_path, img=out)


def compare_vimeo90k_imgx3(src_root, dst_root, name_list, experiment_list, crf_list):
    assert len(experiment_list) == 2
    for name in name_list:
        name_1, name_2 = name.split('/')
        for crf in crf_list:
            h, w, c = 256, 448, 3
            out = np.zeros((1 * h, 3 * w, c), dtype=np.uint8)

            # ground truth
            path = osp.join(src_root, 'GT', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, 'GT', position='tl')
            out[0*h:1*h, 0*w:1*w, :] = img

            # baseline
            path = osp.join(src_root, experiment_list[0], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, 'Baseline', position='tl')
            out[0*h:1*h, 1*w:2*w, :] = img

            # crf 18
            path = osp.join(src_root, experiment_list[1], f'Vimeo90k_crf{crf}_frame', name_1, name_2, 'im4.png')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = draw_text(img, 'Joint Training', position='tl')
            out[0*h:1*h, 2*w:3*w, :] = img

            dst_path = osp.join(dst_root, f'{name_1}_{name_2}_crf{crf}.png')
            cv2.imwrite(dst_path, img=out)


if __name__ == '__main__':
    src_root = '/home/xiyang/data0/datasets/ReCp/isr_results'
    dst_root = '/home/xiyang/data0/datasets/ReCp/compare_isr_results'
    name_list = ['00001/0266', '00001/0268', '00002/0010', '00051/0076']
    crf_list = [18, 23, 28, 33]
    experiment_list = [
        '001_MSRResNet_x2_f64b16_Vimeo90k_250k_B16G1_wandb/visualization',
        '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf18/visualization',
        '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf23/visualization',
        '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf28/visualization',
        '001_MSRResNet_EncoderDecoder_x2_Vimeo90k_250k_crf33/visualization'
    ]
    compare_vimeo90k_imgx6(src_root, dst_root, name_list, experiment_list, crf_list)

    src_root = '/home/xiyang/data0/datasets/ReCp/vsr_results'
    dst_root = '/home/xiyang/data0/datasets/ReCp/compare_vsr_results'
    name_list = ['00001/0266', '00001/0268', '00002/0010', '00051/0076']
    crf_list = [19, 23, 27, 31]
    experiment_list = [
        'BasicVSR_x2_nf64nb10_Vimeo90k_250k/visualization',
        'BasicVSR_x2_nf64nb10_Vimeo90k_250k_crf19/visualization',
        'BasicVSR_x2_nf64nb10_Vimeo90k_250k_crf23/visualization',
        'BasicVSR_x2_nf64nb10_Vimeo90k_250k_crf27/visualization',
        'BasicVSR_x2_nf64nb10_Vimeo90k_250k_crf31/visualization'
    ]
    compare_vimeo90k_imgx6(src_root, dst_root, name_list, experiment_list, crf_list)

    src_root = '/home/xiyang/data0/datasets/ReCp/mqp_results'
    dst_root = '/home/xiyang/data0/datasets/ReCp/compare_mqp_results'
    name_list = ['00001/0266', '00001/0268', '00002/0010', '00051/0076']
    crf_list = [19, 23, 27, 31]
    experiment_list = [
        'MSRResNet_x2_Plain_Vimeo90k_250k',
        'MSRResNet_x2_Joint_Vimeo90k_250k'
    ]
    compare_vimeo90k_imgx3(src_root, dst_root, name_list, experiment_list, crf_list)
