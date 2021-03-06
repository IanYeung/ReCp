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
class MultiFrameSRCompressModel(BaseModel):
    """SR+Compress model for compression aware image super-resolution."""

    def __init__(self, opt):
        super(MultiFrameSRCompressModel, self).__init__(opt)

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

        if train_opt.get('ssim_opt_sr'):
            self.cri_ssim_sr = build_loss(train_opt['ssim_opt_sr']).to(self.device)
        else:
            self.cri_ssim_sr = None

        if train_opt.get('perceptual_opt_sr'):
            self.cri_perceptual_sr = build_loss(train_opt['perceptual_opt_sr']).to(self.device)
        else:
            self.cri_perceptual_sr = None

        if self.cri_pix_sr is None and self.cri_ssim_sr is None and self.cri_perceptual_sr is None:
            raise ValueError('Both pixel and perceptual losses for super-resolution network are None.')

        if train_opt.get('pixel_opt_cp'):
            self.cri_pix_cp = build_loss(train_opt['pixel_opt_cp']).to(self.device)
        else:
            self.cri_pix_cp = None

        if train_opt.get('ssim_opt_cp'):
            self.cri_ssim_cp = build_loss(train_opt['ssim_opt_cp']).to(self.device)
        else:
            self.cri_ssim_cp = None

        if train_opt.get('perceptual_opt_cp'):
            self.cri_perceptual_cp = build_loss(train_opt['perceptual_opt_cp']).to(self.device)
        else:
            self.cri_perceptual_cp = None

        if self.cri_pix_cp is None and self.cri_ssim_sr is None and self.cri_perceptual_cp is None:
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

    def _apply_loss_framewise(self, loss, x, y):
        assert x.shape[1] == y.shape[1]
        sum_loss = 0.
        for i in range(x.shape[1]):
            sum_loss += loss(x[:, i, :, :, :], y[:, i, :, :, :])
        return sum_loss

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
        # ssim loss
        if self.cri_ssim_sr:
            l_ssim_sr = self._apply_loss_framewise(self.cri_ssim_sr, self.output_sr, self.gt)
            l_total += l_ssim_sr
            loss_dict['l_ssim_sr'] = l_ssim_sr
        if self.cri_ssim_cp:
            l_ssim_cp = self._apply_loss_framewise(self.cri_ssim_cp, self.output_cp, self.gt)
            l_total += l_ssim_cp
            loss_dict['l_ssim_cp'] = l_ssim_cp
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
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_imgs = tensor2img(visuals['result_sr'])
            cp_imgs = tensor2img(visuals['result_cp'])
            if 'gt' in visuals:
                gt_imgs = tensor2img(visuals['gt'])
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
                        'im{idx}' + f'_sr_{current_iter:08d}.png')
                else:
                    clip_name_part1, clip_name_part2 = clip_name.split('_')
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_sr_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_sr.png')
                for sr_img_idx, sr_img in zip(val_data['frame_list'], sr_imgs):
                    imwrite(sr_img, save_img_path.format(idx=sr_img_idx.item()))

                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, clip_name,
                        'im{idx}' + f'_cp_{current_iter:08d}.png')
                else:
                    clip_name_part1, clip_name_part2 = clip_name.split('_')
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_cp_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, clip_name_part1, clip_name_part2,
                            'im{idx}' + f'_cp.png')
                for cp_img_idx, cp_img in zip(val_data['frame_list'], cp_imgs):
                    imwrite(cp_img, save_img_path.format(idx=cp_img_idx.item()))

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])

                for name, opt_ in opt_metric.items():
                    metric_results_ = [
                        calculate_metric(dict(img1=sr, img2=gt), opt_)
                        for sr, gt in zip(sr_imgs, gt_imgs)
                    ]
                    self.metric_results_sr[name] += torch.tensor(
                        sum(metric_results_) / len(metric_results_))

                for name, opt_ in opt_metric.items():
                    metric_results_ = [
                        calculate_metric(dict(img1=cp, img2=gt), opt_)
                        for cp, gt in zip(cp_imgs, gt_imgs)
                    ]
                    self.metric_results_cp[name] += torch.tensor(
                        sum(metric_results_) / len(metric_results_))
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
        # dim: n,t,c,h,w
        t = self.lq.shape[1]
        assert (t == self.gt.shape[1] and t == self.output_sr.shape[1] and t == self.output_cp.shape[1])
        lq = self.lq.detach().cpu().squeeze(0)
        gt = self.gt.detach().cpu().squeeze(0)
        result_sr = self.output_sr.detach().cpu().squeeze(0)
        result_cp = self.output_cp.detach().cpu().squeeze(0)
        return dict(
            lq=[lq[i] for i in range(t)],
            gt=[gt[i] for i in range(t)],
            result_sr=[result_sr[i] for i in range(t)],
            result_cp=[result_cp[i] for i in range(t)]
        )

    def save(self, epoch, current_iter):
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_cp, 'net_cp', current_iter)
        self.save_training_state(epoch, current_iter)
