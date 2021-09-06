import math
import random

import torch
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
class SingleFrameSRCompressModel(BaseModel):
    """SR+Compress model for compression aware image super-resolution."""

    def __init__(self, opt):
        super(SingleFrameSRCompressModel, self).__init__(opt)

        # define network
        self.net_sr = build_network(opt['network_sr'])
        self.net_cp = build_network(opt['network_cp'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.net_cp = self.model_to_device(self.net_cp)
        self.print_network(self.net_sr)
        self.print_network(self.net_cp)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_sr', None)
        if load_path is not None:
            self.load_network(self.net_sr, load_path,
                              self.opt['path'].get('strict_load_sr', True))
        load_path = self.opt['path'].get('pretrain_network_cp', None)
        if load_path is not None:
            self.load_network(self.net_cp, load_path,
                              self.opt['path'].get('strict_load_cp', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_sr.train()
        self.net_cp.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt_sr'):
            self.cri_pix_sr = build_loss(train_opt['pixel_opt_sr']).to(self.device)
        else:
            self.cri_pix_sr = None

        if train_opt.get('perceptual_opt_sr'):
            self.cri_perceptual_sr = build_loss(train_opt['perceptual_opt_sr']).to(self.device)
        else:
            self.cri_perceptual_sr = None

        if self.cri_pix_sr is None and self.cri_perceptual_sr is None:
            raise ValueError('Both pixel and perceptual losses for super-resolution network are None.')

        if train_opt.get('pixel_opt_cp'):
            self.cri_pix_cp = build_loss(train_opt['pixel_opt_cp']).to(self.device)
        else:
            self.cri_pix_cp = None

        if train_opt.get('perceptual_opt_cp'):
            self.cri_perceptual_cp = build_loss(train_opt['perceptual_opt_cp']).to(self.device)
        else:
            self.cri_perceptual_cp = None

        if self.cri_pix_cp is None and self.cri_perceptual_cp is None:
            raise ValueError('Both pixel and perceptual losses for compression network are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_sr.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_cp.named_parameters():
            v.requires_grad = False
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
        self.output_sr = self.net_sr(self.lq)
        self.output_cp = self.net_cp(self.output_sr)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix_sr:
            l_pix_sr = self.cri_pix_sr(self.output_sr, self.gt)
            l_total += l_pix_sr
            loss_dict['l_pix_sr'] = l_pix_sr
        if self.cri_pix_cp:
            l_pix_cp = self.cri_pix_cp(self.output_cp, self.gt)
            l_total += l_pix_cp
            loss_dict['l_pix_cp'] = l_pix_cp
        # perceptual loss
        if self.cri_perceptual_sr:
            l_percep_sr, l_style_sr = self.cri_perceptual_sr(self.output_sr, self.gt)
            if l_percep_sr is not None:
                l_total += l_percep_sr
                loss_dict['l_percep_sr'] = l_percep_sr
            if l_style_sr is not None:
                l_total += l_style_sr
                loss_dict['l_style_sr'] = l_style_sr
        if self.cri_perceptual_cp:
            l_percep_cp, l_style_cp = self.cri_perceptua_cp(self.output_cp, self.gt)
            if l_percep_cp is not None:
                l_total += l_percep_cp
                loss_dict['l_percep_sr'] = l_percep_cp
            if l_style_cp is not None:
                l_total += l_style_cp
                loss_dict['l_style_sr'] = l_style_cp

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_sr.eval()
        self.net_cp.eval()
        with torch.no_grad():
            self.output_sr = self.net_sr(self.lq)
            self.output_cp = self.net_cp(self.output_sr)
        self.net_sr.train()
        self.net_cp.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results_sr = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_cp = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            clip_name = val_data['key'][0].replace('/', '_')
            img_name = 'im4'
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result_sr']])
            cp_img = tensor2img([visuals['result_cp']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output_sr
            del self.output_cp
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'sr_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'cp_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["name"]}.png')
                imwrite(cp_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_sr = dict(img1=sr_img, img2=gt_img)
                    self.metric_results_sr[name] += calculate_metric(metric_data_sr, opt_)
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_cp = dict(img1=cp_img, img2=gt_img)
                    self.metric_results_cp[name] += calculate_metric(metric_data_cp, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {clip_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results_sr.keys():
                self.metric_results_sr[metric] /= (idx + 1)
            for metric in self.metric_results_cp.keys():
                self.metric_results_cp[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name} SR\n'
        for metric, value in self.metric_results_sr.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        log_str = f'Validation {dataset_name} CP\n'
        for metric, value in self.metric_results_cp.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results_sr.items():
                tb_logger.add_scalar(f'metrics_sr/{metric}', value, current_iter)
            for metric, value in self.metric_results_cp.items():
                tb_logger.add_scalar(f'metrics_cp/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result_sr'] = self.output_sr.detach().cpu()
        out_dict['result_cp'] = self.output_cp.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_cp, 'net_cp', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class SingleFrameSRMultiQPModel(BaseModel):
    """SR+Compress model for compression aware image super-resolution."""

    def __init__(self, opt):
        super(SingleFrameSRMultiQPModel, self).__init__(opt)

        # define network
        self.net_sr = build_network(opt['network_sr'])
        self.net_cp = build_network(opt['network_cp'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.net_cp = self.model_to_device(self.net_cp)
        self.print_network(self.net_sr)
        self.print_network(self.net_cp)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_sr', None)
        if load_path is not None:
            self.load_network(self.net_sr, load_path,
                              self.opt['path'].get('strict_load_sr', True))
        load_path = self.opt['path'].get('pretrain_network_cp', None)
        if load_path is not None:
            self.load_network(self.net_cp, load_path,
                              self.opt['path'].get('strict_load_cp', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_sr.train()
        self.net_cp.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt_sr'):
            self.cri_pix_sr = build_loss(train_opt['pixel_opt_sr']).to(self.device)
        else:
            self.cri_pix_sr = None
        if train_opt.get('perceptual_opt_sr'):
            self.cri_perceptual_sr = build_loss(train_opt['perceptual_opt_sr']).to(self.device)
        else:
            self.cri_perceptual_sr = None
        # if self.cri_pix_sr is None and self.cri_perceptual_sr is None:
        #     raise ValueError('Both pixel and perceptual losses for super-resolution network are None.')

        if train_opt.get('pixel_opt_cp'):
            self.cri_pix_cp = build_loss(train_opt['pixel_opt_cp']).to(self.device)
        else:
            self.cri_pix_cp = None
        if train_opt.get('perceptual_opt_cp'):
            self.cri_perceptual_cp = build_loss(train_opt['perceptual_opt_cp']).to(self.device)
        else:
            self.cri_perceptual_cp = None
        # if self.cri_pix_cp is None and self.cri_perceptual_cp is None:
        #     raise ValueError('Both pixel and perceptual losses for compression network are None.')

        if train_opt.get('rate_opt'):
            self.cri_bpp = build_loss(train_opt['rate_opt']).to(self.device)
        else:
            self.cri_bpp = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_params = []
        for k, v in self.net_sr.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_cp.named_parameters():
            if not k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer c
        optim_params = []
        for k, v in self.net_cp.named_parameters():
            if not k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_c'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_c = torch.optim.Adam(optim_params,
                                                **train_opt['optim_c'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_c)

        # optimizer a
        optim_params = []
        for k, v in self.net_cp.named_parameters():
            if k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_a'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_a = torch.optim.Adam(optim_params,
                                                **train_opt['optim_a'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_a)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()

        self.output_sr = self.net_sr(self.lq)
        self.output_cp, self.likehihoods = self.net_cp(self.output_sr, qp=random.randint(15, 30), training=True)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix_sr:
            l_pix_sr = self.cri_pix_sr(self.output_sr, self.gt)
            l_total += l_pix_sr
            loss_dict['l_pix_sr'] = l_pix_sr
        if self.cri_pix_cp:
            l_pix_cp = self.cri_pix_cp(self.output_cp, self.gt)
            l_total += l_pix_cp
            loss_dict['l_pix_cp'] = l_pix_cp
        # perceptual loss
        if self.cri_perceptual_sr:
            l_percep_sr, l_style_sr = self.cri_perceptual_sr(self.output_sr, self.gt)
            if l_percep_sr is not None:
                l_total += l_percep_sr
                loss_dict['l_percep_sr'] = l_percep_sr
            if l_style_sr is not None:
                l_total += l_style_sr
                loss_dict['l_style_sr'] = l_style_sr
        if self.cri_perceptual_cp:
            l_percep_cp, l_style_cp = self.cri_perceptua_cp(self.output_cp, self.gt)
            if l_percep_cp is not None:
                l_total += l_percep_cp
                loss_dict['l_percep_sr'] = l_percep_cp
            if l_style_cp is not None:
                l_total += l_style_cp
                loss_dict['l_style_sr'] = l_style_cp
        # rate loss
        if self.cri_bpp:
            N, _, H, W = self.output_sr.shape
            num_pixels = N * H * W
            l_bpp = 0
            for i in range(len(self.likehihoods)):
                l_bpp += self.cri_bpp(self.likehihoods[i], num_pixels)
            # l_bpp = self.cri_bpp(self.likehihoods, num_pixels)
            loss_dict['l_bpp'] = l_bpp
            l_total += l_bpp

        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_c.step()

        aux_loss = self.get_bare_model(self.net_cp).aux_loss()
        loss_dict['l_aux'] = aux_loss
        aux_loss.backward()
        self.optimizer_a.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_sr.eval()
        self.net_cp.eval()
        with torch.no_grad():
            self.output_sr = self.net_sr(self.lq)
            self.output_cp, _ = self.net_cp(self.output_sr, qp=20, training=False)
        self.net_sr.train()
        self.net_cp.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results_sr = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_cp = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            clip_name = val_data['key'][0].replace('/', '_')
            img_name = 'im4'
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result_sr']])
            cp_img = tensor2img([visuals['result_cp']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output_sr
            del self.output_cp
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'sr_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'cp_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["name"]}.png')
                imwrite(cp_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_sr = dict(img1=sr_img, img2=gt_img)
                    self.metric_results_sr[name] += calculate_metric(metric_data_sr, opt_)
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_cp = dict(img1=cp_img, img2=gt_img)
                    self.metric_results_cp[name] += calculate_metric(metric_data_cp, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {clip_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results_sr.keys():
                self.metric_results_sr[metric] /= (idx + 1)
            for metric in self.metric_results_cp.keys():
                self.metric_results_cp[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name} SR\n'
        for metric, value in self.metric_results_sr.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        log_str = f'Validation {dataset_name} CP\n'
        for metric, value in self.metric_results_cp.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results_sr.items():
                tb_logger.add_scalar(f'metrics_sr/{metric}', value, current_iter)
            for metric, value in self.metric_results_cp.items():
                tb_logger.add_scalar(f'metrics_cp/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result_sr'] = self.output_sr.detach().cpu()
        out_dict['result_cp'] = self.output_cp.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_cp, 'net_cp', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class DoubleFrameSRMultiQPModel(BaseModel):
    """SR+Compress model for compression aware image super-resolution."""

    def __init__(self, opt):
        super(DoubleFrameSRMultiQPModel, self).__init__(opt)

        # define network
        self.net_sr = build_network(opt['network_sr'])
        self.net_cp = build_network(opt['network_cp'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.net_cp = self.model_to_device(self.net_cp)
        self.print_network(self.net_sr)
        self.print_network(self.net_cp)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_sr', None)
        if load_path is not None:
            self.load_network(self.net_sr, load_path,
                              self.opt['path'].get('strict_load_sr', True))
        load_path = self.opt['path'].get('pretrain_network_cp', None)
        if load_path is not None:
            self.load_network(self.net_cp, load_path,
                              self.opt['path'].get('strict_load_cp', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_sr.train()
        self.net_cp.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt_sr'):
            self.cri_pix_sr = build_loss(train_opt['pixel_opt_sr']).to(self.device)
        else:
            self.cri_pix_sr = None
        if train_opt.get('perceptual_opt_sr'):
            self.cri_perceptual_sr = build_loss(train_opt['perceptual_opt_sr']).to(self.device)
        else:
            self.cri_perceptual_sr = None
        # if self.cri_pix_sr is None and self.cri_perceptual_sr is None:
        #     raise ValueError('Both pixel and perceptual losses for super-resolution network are None.')

        if train_opt.get('pixel_opt_cp'):
            self.cri_pix_cp = build_loss(train_opt['pixel_opt_cp']).to(self.device)
        else:
            self.cri_pix_cp = None
        if train_opt.get('perceptual_opt_cp'):
            self.cri_perceptual_cp = build_loss(train_opt['perceptual_opt_cp']).to(self.device)
        else:
            self.cri_perceptual_cp = None
        # if self.cri_pix_cp is None and self.cri_perceptual_cp is None:
        #     raise ValueError('Both pixel and perceptual losses for compression network are None.')

        if train_opt.get('rate_opt'):
            self.cri_bpp = build_loss(train_opt['rate_opt']).to(self.device)
        else:
            self.cri_bpp = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_params = []
        for k, v in self.net_sr.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_cp.named_parameters():
            if not k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer c
        optim_params = []
        for k, v in self.net_cp.named_parameters():
            if not k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_c'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_c = torch.optim.Adam(optim_params,
                                                **train_opt['optim_c'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_c)

        # optimizer a
        optim_params = []
        for k, v in self.net_cp.named_parameters():
            if k.endswith(".quantiles"):
                optim_params.append(v)

        optim_type = train_opt['optim_a'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_a = torch.optim.Adam(optim_params,
                                                **train_opt['optim_a'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_a)

    def feed_data(self, data):
        # [B, 2, C, H, W]
        self.lq = data['lq'].to(self.device)
        # [B, 2, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()

        self.prev_frame_lq, self.curr_frame_lq = self.lq[:, 0, :, :, :], self.lq[:, 1, :, :, :]
        self.prev_frame_gt, self.curr_frame_gt = self.gt[:, 0, :, :, :], self.gt[:, 1, :, :, :]

        self.prev_frame_sr = self.net_sr(self.prev_frame_lq)
        self.curr_frame_sr = self.net_sr(self.curr_frame_lq)

        mode = 'intra' if current_iter % 150 == 0 else 'inter'
        self.curr_frame_cp, self.flow, self.likehihoods = \
            self.net_cp(self.curr_frame_sr, self.prev_frame_sr, qp=random.randint(15, 30), training=True, mode=mode)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix_sr:
            l_pix_sr = self.cri_pix_sr(self.curr_frame_sr, self.curr_frame_gt)
            l_total += l_pix_sr
            loss_dict['l_pix_sr'] = l_pix_sr
        if self.cri_pix_cp:
            l_pix_cp = self.cri_pix_cp(self.curr_frame_cp, self.curr_frame_gt)
            l_total += l_pix_cp
            loss_dict['l_pix_cp'] = l_pix_cp
        # perceptual loss
        if self.cri_perceptual_sr:
            l_percep_sr, l_style_sr = self.cri_perceptual_sr(self.curr_frame_sr, self.curr_frame_gt)
            if l_percep_sr is not None:
                l_total += l_percep_sr
                loss_dict['l_percep_sr'] = l_percep_sr
            if l_style_sr is not None:
                l_total += l_style_sr
                loss_dict['l_style_sr'] = l_style_sr
        if self.cri_perceptual_cp:
            l_percep_cp, l_style_cp = self.cri_perceptua_cp(self.curr_frame_cp, self.curr_frame_gt)
            if l_percep_cp is not None:
                l_total += l_percep_cp
                loss_dict['l_percep_sr'] = l_percep_cp
            if l_style_cp is not None:
                l_total += l_style_cp
                loss_dict['l_style_sr'] = l_style_cp
        # rate loss
        if self.cri_bpp:
            N, _, H, W = self.curr_frame_sr.shape
            num_pixels = N * H * W
            l_bpp = 0
            for i in range(len(self.likehihoods)):
                l_bpp += self.cri_bpp(self.likehihoods[i], num_pixels)
            # l_bpp = self.cri_bpp(self.likehihoods, num_pixels)
            loss_dict['l_bpp'] = l_bpp
            l_total += l_bpp

        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_c.step()

        aux_loss = self.get_bare_model(self.net_cp).aux_loss()
        loss_dict['l_aux'] = aux_loss
        aux_loss.backward()
        self.optimizer_a.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_sr.eval()
        self.net_cp.eval()
        with torch.no_grad():
            self.prev_frame_lq, self.curr_frame_lq = self.lq[:, 0, :, :, :], self.lq[:, 1, :, :, :]
            self.prev_frame_gt, self.curr_frame_gt = self.gt[:, 0, :, :, :], self.gt[:, 1, :, :, :]

            self.prev_frame_sr = self.net_sr(self.prev_frame_lq)
            self.curr_frame_sr = self.net_sr(self.curr_frame_lq)

            self.curr_frame_cp, _, _ = \
                self.net_cp(self.curr_frame_sr, self.prev_frame_sr, qp=20, training=False)
        self.net_sr.train()
        self.net_cp.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results_sr = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_cp = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            clip_name = val_data['key'][0].replace('/', '_')
            img_name = 'im4'
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['curr_sr']])
            cp_img = tensor2img([visuals['curr_cp']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['curr_gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.curr_frame_sr
            del self.curr_frame_cp
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'sr_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'sr_{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        f'cp_{img_name}_{current_iter:08d}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name,
                            f'cp_{img_name}_{self.opt["name"]}.png')
                imwrite(cp_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_sr = dict(img1=sr_img, img2=gt_img)
                    self.metric_results_sr[name] += calculate_metric(metric_data_sr, opt_)
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data_cp = dict(img1=cp_img, img2=gt_img)
                    self.metric_results_cp[name] += calculate_metric(metric_data_cp, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {clip_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results_sr.keys():
                self.metric_results_sr[metric] /= (idx + 1)
            for metric in self.metric_results_cp.keys():
                self.metric_results_cp[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name} SR\n'
        for metric, value in self.metric_results_sr.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        log_str = f'Validation {dataset_name} CP\n'
        for metric, value in self.metric_results_cp.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results_sr.items():
                tb_logger.add_scalar(f'metrics_sr/{metric}', value, current_iter)
            for metric, value in self.metric_results_cp.items():
                tb_logger.add_scalar(f'metrics_cp/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['curr_sr'] = self.curr_frame_sr.detach().cpu()
        out_dict['curr_cp'] = self.curr_frame_cp.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['curr_gt'] = self.curr_frame_gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_cp, 'net_cp', current_iter)
        self.save_training_state(epoch, current_iter)
