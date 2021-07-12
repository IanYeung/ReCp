import os
import sys
import os.path as path
import cv2
import glob
import logging
import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.data.data_util import generate_frame_indices, generate_frame_indices_with_scene
from basicsr.utils import mkdir, img2tensor, tensor2img, imread, imwrite, normalize
from basicsr.archs import build_network


logger = logging.getLogger('basicsr')


class SingleFrameEnhancer(object):

    def __init__(self, opt, load_path, device='cuda:0'):
        super(SingleFrameEnhancer, self).__init__()

        self.device = device
        self.net = build_network(opt)
        self._load_network(self.net, load_path)
        self.net = self.net.to(device=device)
        self.net.eval()

    def _load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self._get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def _get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self._get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def forward_on_folder(self, input_folder, save_folder, format='bgr'):
        """
            do inference on a list of image of the input_folder,
            and save the results in the save_folder
            Args:
                input_folder (str): a folder in which input images are stored
                save_folder  (str): a folder to save output images
        """
        mkdir(save_folder)
        img_list = sorted(glob.glob(path.join(input_folder, '*.png')))
        logger.info('processing images from {}...'.format(input_folder))

        for img_path in img_list:
            logger.info('processing: {}'.format(img_path))
            img = cv2.imread(img_path)
            img_name = path.basename(img_path)

            sr_img = self.forward_single(img, format=format)

            cv2.imwrite(path.join(save_folder, img_name), sr_img)

        logger.info('sr processing done!')

    def forward_single(self, img, format='bgr'):
        """
            do inference on a single input image
            Args:
                img (np.uint8): input image with BGR format
            Returns:
                out (np.uint8): output image with BGR format
        """
        if format == 'bgr':
            # BGR
            with torch.no_grad():
                inp = img2tensor(normalize(img))
                inp = inp.unsqueeze(0).to(self.device)
                out = self.net(inp)
                out = tensor2img(out)
            return out
        else:
            # Y/GRAY
            with torch.no_grad():
                inp = img2tensor(normalize(np.expand_dims(img, axis=2)))
                inp = inp.unsqueeze(0).to(self.device)
                out = self.net(inp)
                out = tensor2img(out)
            return out


# class MultiFrameEnhancer(object):
#
#     def __init__(self, opt, load_path, device_id=0):
#         super(MultiFrameEnhancer, self).__init__()
#         self.device_id = device_id
#         self.net = build_network(opt)
#         self._load_network(self.net, load_path)
#
#         self.net = self.net.cuda(device=device_id)
#         self.net.eval()
#
#     def _load_network(self, net, load_path, strict=True, param_key='params'):
#         """Load network.
#
#         Args:
#             load_path (str): The path of networks to be loaded.
#             net (nn.Module): Network.
#             strict (bool): Whether strictly loaded.
#             param_key (str): The parameter key of loaded network. If set to
#                 None, use the root 'path'.
#                 Default: 'params'.
#         """
#         net = self._get_bare_model(net)
#         logger.info(
#             f'Loading {net.__class__.__name__} model from {load_path}.')
#         load_net = torch.load(
#             load_path, map_location=lambda storage, loc: storage)
#         if param_key is not None:
#             load_net = load_net[param_key]
#         # remove unnecessary 'module.'
#         for k, v in deepcopy(load_net).items():
#             if k.startswith('module.'):
#                 load_net[k[7:]] = v
#                 load_net.pop(k)
#         self._print_different_keys_loading(net, load_net, strict)
#         net.load_state_dict(load_net, strict=strict)
#
#     def _get_bare_model(self, net):
#         """Get bare model, especially under wrapping with
#         DistributedDataParallel or DataParallel.
#         """
#         if isinstance(net, (DataParallel, DistributedDataParallel)):
#             net = net.module
#         return net
#
#     def _print_different_keys_loading(self, crt_net, load_net, strict=True):
#         """Print keys with differnet name or different size when loading models.
#
#         1. Print keys with differnet names.
#         2. If strict=False, print the same key but with different tensor size.
#             It also ignore these keys with different sizes (not load).
#
#         Args:
#             crt_net (torch model): Current network.
#             load_net (dict): Loaded network.
#             strict (bool): Whether strictly loaded. Default: True.
#         """
#         crt_net = self._get_bare_model(crt_net)
#         crt_net = crt_net.state_dict()
#         crt_net_keys = set(crt_net.keys())
#         load_net_keys = set(load_net.keys())
#
#         if crt_net_keys != load_net_keys:
#             logger.warning('Current net - loaded net:')
#             for v in sorted(list(crt_net_keys - load_net_keys)):
#                 logger.warning(f'  {v}')
#             logger.warning('Loaded net - current net:')
#             for v in sorted(list(load_net_keys - crt_net_keys)):
#                 logger.warning(f'  {v}')
#
#         # check the size for the same keys
#         if not strict:
#             common_keys = crt_net_keys & load_net_keys
#             for k in common_keys:
#                 if crt_net[k].size() != load_net[k].size():
#                     logger.warning(
#                         f'Size different, ignore [{k}]: crt_net: '
#                         f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
#                     load_net[k + '.ignore'] = load_net.pop(k)
#
#     def _scene_detect_folder(self, input_folder, threshold=10000000):
#
#         idx_list = []
#         frm_paths = sorted(glob.glob(path.join(input_folder, '*')))
#         length = len(frm_paths)
#
#         idx_list.append(0)
#         for (idx, frm_path) in enumerate(frm_paths):
#             frame_BGR = cv2.imread(frm_path)
#             frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)
#             histo_curr = cv2.calcHist([frame_HSV], [0], None, [256], [0.0, 255.0])
#
#             if idx == 0:
#                 histo_prev = histo_curr
#
#             diff = histo_curr - histo_prev
#             value = np.mean(np.power(diff, 2))
#
#             if value > threshold:
#                 idx_list.append(idx)
#
#             histo_prev = histo_curr
#         idx_list.append(length)
#
#         return idx_list
#
#     def forward_on_folder(self, input_folder, save_folder):
#         '''
#             do inference on a list of image of the input_folder,
#             and save the results in the save_folder
#             Args:
#                 input_folder (str): a folder in which input images are stored
#                 save_folder  (str): a folder to save output images
#         '''
#         mkdir(save_folder)
#         img_list = sorted(glob.glob(path.join(input_folder, '*.png')))
#         scene_list = self._scene_detect_folder(input_folder)
#
#         length = len(img_list)
#         logger.info('processing images from {}...'.format(input_folder))
#
#         for img_idx, img_path in enumerate(img_list):
#             img_name = path.basename(img_path)
#             logger.info('Processing: {}'.format(img_name))
#             select_idx = generate_frame_indices_with_scene(img_idx, length, 3, scene_list, padding='replicate')
#             img_l = []
#             for idx in select_idx:
#                 inp_path = path.join(input_folder, '{:05d}.png'.format(idx + 1))
#                 img_l.append(imread(inp_path, flag='unchanged'))
#             sr_img = self.forward_single(img_l)
#
#             # save img
#             cv2.imwrite(path.join(save_folder, img_name), sr_img)
#
#         logger.info('sr processing done!')
#
#     def forward_single(self, imgs):
#         with torch.no_grad():
#             imgs = img2tensor(normalize(imgs))
#             inp = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)
#             out = self.net(inp)
#             out = tensor2img(out)
#             return out


class MultiFrameEnhancer(object):
    def __init__(self, opt, load_path, nframes=5, device='cuda:0'):
        super(MultiFrameEnhancer, self).__init__()
        self.nframes = nframes
        self.device = device

        self.net = build_network(opt)
        self._load_network(self.net, load_path)
        self.net = self.net.to(device=device)
        self.net.eval()

        # for cache multiframe input
        self.input_cache = []
        self.input_pool_len = 0

    def _load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self._get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def _get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self._get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def clear_input_cache(self):
        self.input_cache.clear()
        self.input_pool_len = 0

    def get_pool_length(self):
        return self.input_pool_len

    def sequence_input_pool(self, img):
        """
            store input image as gpu tensor buffer for accelerating
            use before invoke FUNC. [forward_sequence]
            Args:
                img (np.uint8): input image with RGB HWC format
        """
        with torch.no_grad():
            tensor = torch.from_numpy(img)
            tensor = tensor.to(device=self.device)
            tensor = tensor.permute(2, 0, 1)  # HWC2CHW
            tensor = tensor.float()
            tensor = tensor / 255.
            self.input_cache.append(tensor)

        self.input_pool_len += 1

    def forward_sequence(self, idxs):
        """
            do inference on a single input image
            Args:
                idxs (list<int>): input image idx
            Returns:
                out (np.uint8): output image with RGB format
        """
        with torch.no_grad():
            input = torch.stack([self.input_cache[v] for v in idxs], dim=0)  # TCHW
            input = input.unsqueeze(0).tp(device=self.device)  # NTCHW
            input[input < (16. / 255.)] = 16. / 255.
            # frames = torch.zeros(1, self.nframes, 3, 1088, 1920).cuda(device=self.device_id)
            # frames[:, :, 0, :, :] =  16. / 255.
            # frames[:, :, 1, :, :] = 128. / 255.
            # frames[:, :, 2, :, :] = 128. / 255.
            # frames[:, :, :, 4:-4, :] = input

            out = self.net(input)
            out = out * 255.
            out = out.to(torch.uint8).squeeze()
            out = out.permute(1, 2, 0)
            out = out.cpu().numpy()

            # out = out[4:-4, :, :]

        return out


if __name__ == '__main__':

    opt = {'type': 'MSRResNet', 'num_feat': 64, 'num_block': 16, 'num_in_ch': 3, 'num_out_ch': 3, 'upscale': 2}
    load_path = '/home/xiyang/Projects/ReCp/experiments/pretrained_models/MSRResNet_x2.pth'
    device = 'cuda:0'
    enhancer = SingleFrameEnhancer(opt, load_path, device)

    read_root = '/home/xiyang/Datasets/test/test_images/lq_images'
    save_root = '/home/xiyang/Datasets/test/test_images/hq_images'
    mkdir(save_root)
    img_paths = sorted(glob.glob(path.join(read_root, '*')))
    for img_path in img_paths:
        print(img_path)
        lr_img = cv2.imread(img_path)
        sr_img = enhancer.forward_single(lr_img)
        cv2.imwrite(path.join(save_root, path.basename(img_path)), img=sr_img)
