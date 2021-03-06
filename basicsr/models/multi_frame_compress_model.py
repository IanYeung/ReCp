import torch
from copy import deepcopy
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class MultiFrameCompressModel(BaseModel):
    """Compress model for compression modeling."""

    def __init__(self, opt):
        super(MultiFrameCompressModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            clip_name = val_data['key'][0].replace('/', '_')
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_imgs = tensor2img(visuals['result'])
            if 'gt' in visuals:
                gt_imgs = tensor2img(visuals['gt'])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             clip_name,
                                             'im{idx}' + f'_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        clip_name_part1, clip_name_part2 = clip_name.split('_')
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_{self.opt["val"]["suffix"]}.png')
                    else:
                        clip_name_part1, clip_name_part2 = clip_name.split('_')
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}.png')

                for sr_img_idx, sr_img in zip(val_data['frame_list'], sr_imgs):
                    imwrite(sr_img, save_img_path.format(idx=sr_img_idx.item()))

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_results_ = [
                        calculate_metric(dict(img1=sr, img2=gt), opt_)
                        for sr, gt in zip(sr_imgs, gt_imgs)
                    ]
                    self.metric_results[name] += torch.tensor(
                        sum(metric_results_) / len(metric_results_))
            pbar.update(1)
            pbar.set_description(f'Test {clip_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        # dim: n,t,c,h,w
        t = self.lq.shape[1]
        assert (t == self.gt.shape[1] and t == self.output.shape[1])
        lq = self.lq.detach().cpu().squeeze(0)
        gt = self.gt.detach().cpu().squeeze(0)
        result = self.output.detach().cpu().squeeze(0)
        return dict(
            lq=[lq[i] for i in range(t)],
            gt=[gt[i] for i in range(t)],
            result=[result[i] for i in range(t)])

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class MultiFrameCondCompressModel(BaseModel):
    """Compress model for compression modeling."""

    def __init__(self, opt):
        super(MultiFrameCondCompressModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'cd' in data:
            self.cd = data['cd'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g((self.lq, self.cd))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g((self.lq, self.cd))
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            clip_name = val_data['key'][0].replace('/', '_')
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_imgs = tensor2img(visuals['result'])
            if 'gt' in visuals:
                gt_imgs = tensor2img(visuals['gt'])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             clip_name,
                                             'im{idx}' + f'_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        clip_name_part1, clip_name_part2 = clip_name.split('_')
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_{self.opt["val"]["suffix"]}.png')
                    else:
                        clip_name_part1, clip_name_part2 = clip_name.split('_')
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}.png')

                for sr_img_idx, sr_img in zip(val_data['frame_list'], sr_imgs):
                    imwrite(sr_img, save_img_path.format(idx=sr_img_idx.item()))

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_results_ = [
                        calculate_metric(dict(img1=sr, img2=gt), opt_)
                        for sr, gt in zip(sr_imgs, gt_imgs)
                    ]
                    self.metric_results[name] += torch.tensor(
                        sum(metric_results_) / len(metric_results_))
            pbar.update(1)
            pbar.set_description(f'Test {clip_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        # dim: n,t,c,h,w
        t = self.lq.shape[1]
        assert (t == self.gt.shape[1] and t == self.output.shape[1])
        lq = self.lq.detach().cpu().squeeze(0)
        gt = self.gt.detach().cpu().squeeze(0)
        result = self.output.detach().cpu().squeeze(0)
        return dict(
            lq=[lq[i] for i in range(t)],
            gt=[gt[i] for i in range(t)],
            result=[result[i] for i in range(t)])

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)