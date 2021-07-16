import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from basicsr.utils import mkdir


def sf_results():
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    LineWidth = 2

    PSNR, SSIM = True, True

    if PSNR:
        bpp = [0.00053461, 0.00036419, 0.00025088, 0.00017527]
        bps = [1.53282124, 1.04421109, 0.71932855, 0.50254236]
        psnr = [31.5574, 31.0808, 30.2522, 28.9882]
        sr_baseline_psnr, = plt.plot(bps, psnr, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.00058057, 0.00038518, 0.00025965, 0.00017888]
        bps = [1.66460327, 1.10438410, 0.74445437, 0.51288960]
        psnr = [32.1913, 31.6979, 30.7606, 29.3301]
        sr_cp_crf18_psnr, = plt.plot(bps, psnr, "g-o", linewidth=LineWidth, label='Joint Training (CRF18)')

        bpp = [0.00062365, 0.00040931, 0.00027154, 0.00018400]
        bps = [1.78813855, 1.17356511, 0.77854972, 0.52755404]
        psnr = [32.0263, 31.6678, 30.8174, 29.3943]
        sr_cp_crf23_psnr, = plt.plot(bps, psnr, "c-o", linewidth=LineWidth, label='Joint Training (CRF23)')

        bpp = [0.00068612, 0.00044797, 0.00029307, 0.00019387]
        bps = [1.96723543, 1.28443149, 0.84028947, 0.55585360]
        psnr = [31.5432, 31.3855, 30.7653, 29.4650]
        sr_cp_crf28_psnr, = plt.plot(bps, psnr, "r-o", linewidth=LineWidth, label='Joint Training (CRF28)')

        bpp = [0.00077178, 0.00050484, 0.00033070, 0.00021531]
        bps = [2.21285680, 1.44748113, 0.94818699, 0.61732947]
        psnr = [30.3129, 30.3310, 30.1172, 29.3276]
        sr_cp_crf33_psnr, = plt.plot(bps, psnr, "y-o", linewidth=LineWidth, label='Joint Training (CRF33)')

        plt.legend(handles=[sr_baseline_psnr,
                            sr_cp_crf18_psnr,
                            sr_cp_crf23_psnr,
                            sr_cp_crf28_psnr,
                            sr_cp_crf33_psnr], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('PSNR')
        plt.title('Vimeo90k Dataset')
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_isr_results/bd_psnr.png')
        plt.clf()

    if SSIM:
        bpp = [0.00053461, 0.00036419, 0.00025088, 0.00017527]
        bps = [1.53282124, 1.04421109, 0.71932855, 0.50254236]
        msssim = [0.968068, 0.963321, 0.953891, 0.935410]
        sr_baseline_msssim, = plt.plot(bps, msssim, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.00058057, 0.00038518, 0.00025965, 0.00017888]
        bps = [1.66460327, 1.10438410, 0.74445437, 0.51288960]
        msssim = [0.969375, 0.965291, 0.956277, 0.937465]
        sr_cp_crf18_msssim, = plt.plot(bps, msssim, "g-o", linewidth=LineWidth, label='Joint Training (CRF18)')

        bpp = [0.00062365, 0.00040931, 0.00027154, 0.00018400]
        bps = [1.78813855, 1.17356511, 0.77854972, 0.52755404]
        msssim = [0.968106, 0.965256, 0.957174, 0.938596]
        sr_cp_crf23_msssim, = plt.plot(bps, msssim, "c-o", linewidth=LineWidth, label='Joint Training (CRF23)')

        bpp = [0.00068612, 0.00044797, 0.00029307, 0.00019387]
        bps = [1.96723543, 1.28443149, 0.84028947, 0.55585360]
        msssim = [0.964872, 0.963608, 0.957330, 0.940146]
        sr_cp_crf28_msssim, = plt.plot(bps, msssim, "r-o", linewidth=LineWidth, label='Joint Training (CRF28)')

        bpp = [0.00077178, 0.00050484, 0.00033070, 0.00021531]
        bps = [2.21285680, 1.44748113, 0.94818699, 0.61732947]
        msssim = [0.956186, 0.956095, 0.952903, 0.940634]
        sr_cp_crf33_msssim, = plt.plot(bps, msssim, "y-o", linewidth=LineWidth, label='Joint Training (CRF33)')

        plt.legend(handles=[sr_baseline_msssim,
                            sr_cp_crf18_msssim,
                            sr_cp_crf23_msssim,
                            sr_cp_crf28_msssim,
                            sr_cp_crf33_msssim], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('MS-SSIM')
        plt.title('Vimeo90k Dataset')
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_isr_results/bd_ssim.png')
        plt.clf()


def mf_results():
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    LineWidth = 2

    PSNR, SSIM = True, True

    if PSNR:
        bpp = [0.00019041, 0.00013092, 0.00009249, 0.00006670]
        bps = [0.54594702, 0.37538160, 0.26517825, 0.19123745]
        psnr = [31.9541, 31.6547, 31.1728, 30.4531]  # PSNR model
        # psnr = [31.9541, 31.6547, 31.1728, 30.4531]  # MS-SSIM model
        sr_baseline_psnr, = plt.plot(bps, psnr, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.00020363, 0.00013716, 0.00009537, 0.00006809]
        bps = [0.58384865, 0.39326338, 0.27344328, 0.19523968]
        # psnr = [32.6490, 32.3192, 31.7551, 30.9100]  # PSNR model
        psnr = [32.6006, 32.2756, 31.7261, 30.8992]  # MS-SSIM model
        sr_cp_crf19_psnr, = plt.plot(bps, psnr, "g-o", linewidth=LineWidth, label='Joint Training (CRF19)')

        bpp = [0.00021290, 0.00014201, 0.00009790, 0.00006942]
        bps = [0.61041574, 0.40717726, 0.28069740, 0.19903997]
        # psnr = [32.5900, 32.3051, 31.7778, 30.9493]  # PSNR model
        psnr = [32.5341, 32.2469, 31.7311, 30.9211]  # MS-SSIM model
        sr_cp_crf23_psnr, = plt.plot(bps, psnr, "c-o", linewidth=LineWidth, label='Joint Training (CRF23)')

        bpp = [0.00022847, 0.00015061, 0.00010261, 0.00007198]
        bps = [0.65505649, 0.43181749, 0.29421737, 0.20637096]
        # psnr = [32.3802, 32.1775, 31.7405, 30.9757]  # PSNR model
        psnr = [32.2977, 32.0902, 31.6615, 30.9148]  # MS-SSIM model
        sr_cp_crf27_psnr, = plt.plot(bps, psnr, "r-o", linewidth=LineWidth, label='Joint Training (CRF27)')

        bpp = [0.00025039, 0.00016297, 0.00010957, 0.00007582]
        bps = [0.71791289, 0.46725643, 0.31417159, 0.21739337]
        # psnr = [31.9585, 31.8515, 31.5441, 30.9121]  # PSNR model
        psnr = [31.7792, 31.6820, 31.4031, 30.8036]  # MS-SSIM model
        sr_cp_crf31_psnr, = plt.plot(bps, psnr, "y-o", linewidth=LineWidth, label='Joint Training (CRF31)')

        plt.legend(handles=[sr_baseline_psnr,
                            sr_cp_crf19_psnr,
                            sr_cp_crf23_psnr,
                            sr_cp_crf27_psnr,
                            sr_cp_crf31_psnr], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('PSNR')
        plt.title('Vimeo90k Dataset')
        plt.tight_layout()
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_vsr_results/bd_psnr.png')
        plt.clf()

    if SSIM:
        bpp = [0.00019041, 0.00013092, 0.00009249, 0.00006670]
        bps = [0.54594702, 0.37538160, 0.26517825, 0.19123745]
        msssim = [0.970790, 0.968247, 0.963603, 0.955597]  # PSNR model
        # msssim = [0.970790, 0.968247, 0.963603, 0.955597]  # MS-SSIM model
        sr_baseline_msssim, = plt.plot(bps, msssim, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.00020363, 0.00013716, 0.00009537, 0.00006809]
        bps = [0.58384865, 0.39326338, 0.27344328, 0.19523968]
        # msssim = [0.972294, 0.970091, 0.965637, 0.957633]  # PSNR model
        msssim = [0.972673, 0.970441, 0.966120, 0.958377]  # MS-SSIM model
        sr_cp_crf19_msssim, = plt.plot(bps, msssim, "g-o", linewidth=LineWidth, label='Joint Training (CRF19)')

        bpp = [0.00021290, 0.00014201, 0.00009790, 0.00006942]
        bps = [0.61041574, 0.40717726, 0.28069740, 0.19903997]
        # msssim = [0.971901, 0.970067, 0.965966, 0.958193]  # PSNR model
        msssim = [0.972420, 0.970412, 0.966322, 0.958714]  # MS-SSIM model
        sr_cp_crf23_msssim, = plt.plot(bps, msssim, "c-o", linewidth=LineWidth, label='Joint Training (CRF23)')

        bpp = [0.00022847, 0.00015061, 0.00010261, 0.00007198]
        bps = [0.65505649, 0.43181749, 0.29421737, 0.20637096]
        # msssim = [0.970490, 0.969256, 0.965896, 0.958746]  # PSNR model
        msssim = [0.971468, 0.969738, 0.966121, 0.958930]  # MS-SSIM model
        sr_cp_crf27_msssim, = plt.plot(bps, msssim, "r-o", linewidth=LineWidth, label='Joint Training (CRF27)')

        bpp = [0.00025039, 0.00016297, 0.00010957, 0.00007582]
        bps = [0.71791289, 0.46725643, 0.31417159, 0.21739337]
        # msssim = [0.967982, 0.967357, 0.964885, 0.958731]  # PSNR model
        msssim = [0.969895, 0.968226, 0.965009, 0.958491]  # MS-SSIM model
        sr_cp_crf31_msssim, = plt.plot(bps, msssim, "y-o", linewidth=LineWidth, label='Joint Training (CRF31)')

        plt.legend(handles=[sr_baseline_msssim,
                            sr_cp_crf19_msssim,
                            sr_cp_crf23_msssim,
                            sr_cp_crf27_msssim,
                            sr_cp_crf31_msssim], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('MS-SSIM')
        plt.title('Vimeo90k Dataset')
        plt.tight_layout()
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_vsr_results/bd_ssim.png')
        plt.clf()


def qp_results():
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    LineWidth = 2

    PSNR, SSIM = True, True

    if PSNR:
        bpp = [0.57688426, 0.39277123, 0.27477157, 0.19614153]
        psnr = [31.7595, 31.5077, 31.0767, 30.3941]
        sr_plain_psnr, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.63471999, 0.42500515, 0.28463698, 0.19300792]
        psnr = [29.8641, 29.7582, 29.5367, 29.1234]
        sr_joint_psnr, = plt.plot(bpp, psnr, "g-o", linewidth=LineWidth, label='Joint Training')

        plt.legend(handles=[sr_plain_psnr, sr_joint_psnr], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('PSNR')
        plt.title('Vimeo90k Dataset')
        plt.tight_layout()
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_mqp_results/bd_psnr.png')
        plt.clf()

    if SSIM:
        bpp = [0.57688426, 0.39277123, 0.27477157, 0.19614153]
        msssim = [0.969801, 0.967489, 0.963073, 0.955194]
        sr_plain_msssim, = plt.plot(bpp, msssim, "b-*", linewidth=LineWidth, label='Baseline')

        bpp = [0.63471999, 0.42500515, 0.28463698, 0.19300792]
        msssim = [0.966641, 0.963917, 0.958790, 0.949952]
        sr_joint_msssim, = plt.plot(bpp, msssim, "g-o", linewidth=LineWidth, label='Joint Training')

        plt.legend(handles=[sr_plain_msssim, sr_joint_msssim], loc=4)
        plt.grid()
        plt.xlabel('kbps')
        plt.ylabel('MS-SSIM')
        plt.title('Vimeo90k Dataset')
        plt.tight_layout()
        # plt.show()
        plt.savefig('/home/xiyang/data0/datasets/ReCp/compare_mqp_results/bd_ssim.png')
        plt.clf()


if __name__ == '__main__':
    # sf_results()
    # mf_results()
    qp_results()
