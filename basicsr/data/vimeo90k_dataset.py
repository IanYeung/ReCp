import random
import torch
import numpy as np
import decord as de
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import bgr2ycbcr, ycbcr2bgr, rgb2ycbcr, ycbcr2rgb


@DATASET_REGISTRY.register()
class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
                                             img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_flip'],
                              self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KTrainDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KTrainDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
                                             img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_flip'],
                              self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KValidDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KValidDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        img_lqs.append(img_gt)

        img_results = img2tensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KSingleFrameTrainDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KSingleFrameTrainDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)

        # get the LQ frame (im4.png)
        if self.is_lmdb:
            img_lq_path = f'{key}/im4'
        else:
            img_lq_path = self.lq_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)

        # randomly crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_results = augment([img_lq, img_gt], self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lq = img_results[0]
        img_gt = img_results[1]

        # img_lq: (c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KSingleFrameValidDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KSingleFrameValidDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)

        # get the LQ frame (im4.png)
        if self.is_lmdb:
            img_lq_path = f'{key}/im4'
        else:
            img_lq_path = self.lq_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)

        img_results = img2tensor([img_lq, img_gt])
        img_lq = img_results[0]
        img_gt = img_results[1]

        # img_lq: (c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KMultipleGTTrainDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KMultipleGTTrainDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.num_frame = opt['num_frame']
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=-1)
            img_gts.append(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, key)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': self.neighbor_list}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KMultipleGTValidDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KMultipleGTValidDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.num_frame = opt['num_frame']
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=-1)
            img_gts.append(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)

        img_results = img2tensor(img_lqs)
        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': self.neighbor_list}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KDoubleFrameTrainDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDoubleFrameTrainDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        selected_idx = random.randint(1, 6)
        self.neighbor_list = [selected_idx, selected_idx+1]

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=-1)
            img_gts.append(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, key)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[:2], dim=0)
        img_gts = torch.stack(img_results[2:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': self.neighbor_list}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KDoubleFrameValidDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDoubleFrameValidDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        selected_idx = 3
        self.neighbor_list = [selected_idx, selected_idx+1]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=-1)
            img_gts.append(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)
            if img_lq.ndim == 2:
                img_lq = np.expand_dims(img_lq, axis=-1)
            img_lqs.append(img_lq)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)

        img_results = img2tensor(img_lqs)
        img_lqs = torch.stack(img_results[:2], dim=0)
        img_gts = torch.stack(img_results[2:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': self.neighbor_list}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KSingleFrameTrainVideoDataset(data.Dataset):

    def __init__(self, opt):
        super(Vimeo90KSingleFrameTrainVideoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

    def __getitem__(self, index):

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the center GT frame (im4)
        seq_gt_path = self.gt_root / clip / '{}.mp4'.format(seq)
        with open(str(seq_gt_path), 'rb') as f_gt:
            vr_gt = de.VideoReader(f_gt)
        img_gt = vr_gt[len(vr_gt)//2].asnumpy().astype(np.float32) / 255.
        if self.opt['color'] == 'y':
            img_gt = np.expand_dims(rgb2ycbcr(img_gt, y_only=True), axis=-1)

        # get the center LQ frame (im4)
        seq_lq_path = self.lq_root / clip / '{}.mp4'.format(seq)
        with open(str(seq_lq_path), 'rb') as f_lq:
            vr_lq = de.VideoReader(f_lq)
        img_lq = vr_lq[len(vr_lq)//2].asnumpy().astype(np.float32) / 255.
        if self.opt['color'] == 'y':
            img_lq = np.expand_dims(rgb2ycbcr(img_lq, y_only=True), axis=-1)

        # randomly crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, str(seq_gt_path))

        # augmentation - flip, rotate
        img_results = augment([img_lq, img_gt], self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results, bgr2rgb=False)
        img_lq = img_results[0]
        img_gt = img_results[1]

        # img_lq: (c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class Vimeo90KSingleFrameValidVideoDataset(data.Dataset):

    def __init__(self, opt):
        super(Vimeo90KSingleFrameValidVideoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

    def __getitem__(self, index):

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the center GT frame (im4)
        seq_gt_path = self.gt_root / clip / '{}.mp4'.format(seq)
        with open(str(seq_gt_path), 'rb') as f_gt:
            vr_gt = de.VideoReader(f_gt)
        img_gt = vr_gt[len(vr_gt)//2].asnumpy().astype(np.float32) / 255.
        if self.opt['color'] == 'y':
            img_gt = np.expand_dims(rgb2ycbcr(img_gt, y_only=True), axis=-1)

        # get the center LQ frame (im4)
        seq_lq_path = self.lq_root / clip / '{}.mp4'.format(seq)
        with open(str(seq_lq_path), 'rb') as f_lq:
            vr_lq = de.VideoReader(f_lq)
        img_lq = vr_lq[len(vr_lq)//2].asnumpy().astype(np.float32) / 255.
        if self.opt['color'] == 'y':
            img_lq = np.expand_dims(rgb2ycbcr(img_lq, y_only=True), axis=-1)

        img_results = img2tensor([img_lq, img_gt], bgr2rgb=False)
        img_lq = img_results[0]
        img_gt = img_results[1]

        # img_lq: (c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)
