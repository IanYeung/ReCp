import os
import cv2
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from basicsr.utils import mkdir


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff


def obtain_results(model_name):
    metrics_file = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/results/metrics/{}.log'.format(model_name)
    bitrate_file = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/results/bitrate/{}.log'.format(model_name)

    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    psnr = lines[-3].split('PSNR list:')[-1]
    ssim = lines[-2].split('SSIM list:')[-1]
    vmaf = lines[-1].split('VMAF list:')[-1]

    psnr_ = [float(x[2:-1]) for x in psnr[1:-2].split(',')]
    ssim_ = [float(x[2:-1]) for x in ssim[1:-2].split(',')]
    vmaf_ = [float(x[2:-1]) for x in vmaf[1:-2].split(',')]

    with open(bitrate_file, 'r') as f:
        lines = f.readlines()
    bitr = lines[-1].split('kbps list:')[-1]
    bitr_ = [float(x[2:-1]) for x in bitr[1:-2].split(',')]

    return psnr_, ssim_, vmaf_, bitr_


def plot_rd_curve(save_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/plots', save_fig=False):
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    LineWidth = 2

    PSNR, SSIM, VMAF = True, True, True

    model_name_1 = 'MSRResNet_x2_Vimeo90k_250k_Y'
    model_name_2 = 'MSRResNet_EncoderDecoder_Y_x2_Vimeo90k_150k_LossRatio_SR0.1_CP1.0_L1_crf18'

    psnr1, ssim1, vmaf1, bpp1 = obtain_results(model_name_1)
    psnr2, ssim2, vmaf2, bpp2 = obtain_results(model_name_2)

    if PSNR:
        sr_plain_psnr, = plt.plot(bpp1, psnr1, "b-*", linewidth=LineWidth, label='Baseline')
        sr_joint_psnr, = plt.plot(bpp2, psnr2, "g-o", linewidth=LineWidth, label='Joint Training')

        plt.legend(handles=[sr_plain_psnr, sr_joint_psnr], loc=4)
        plt.grid()
        plt.xlabel('bitrate (kbps)')
        plt.ylabel('PSNR')
        plt.title('MCL-JVC Dataset')
        plt.tight_layout()
        if not save_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(save_root, 'rd_curve_psnr.png'))
        plt.clf()
        print('BD-RATE (PSNR): ', BD_RATE(bpp1, psnr1, bpp2, psnr2))

    if SSIM:
        sr_plain_ssim, = plt.plot(bpp1, ssim1, "b-*", linewidth=LineWidth, label='Baseline')
        sr_joint_ssim, = plt.plot(bpp2, ssim2, "g-o", linewidth=LineWidth, label='Joint Training')

        plt.legend(handles=[sr_plain_ssim, sr_joint_ssim], loc=4)
        plt.grid()
        plt.xlabel('bitrate (kbps)')
        plt.ylabel('SSIM')
        plt.title('MCL-JVC Dataset')
        plt.tight_layout()
        if not save_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(save_root, 'rd_curve_ssim.png'))
        plt.clf()
        print('BD-RATE (SSIM): ', BD_RATE(bpp1, ssim1, bpp2, ssim2))

    if VMAF:
        sr_plain_vmaf, = plt.plot(bpp1, vmaf1, "b-*", linewidth=LineWidth, label='Baseline')
        sr_joint_vmaf, = plt.plot(bpp2, vmaf2, "g-o", linewidth=LineWidth, label='Joint Training')

        plt.legend(handles=[sr_plain_vmaf, sr_joint_vmaf], loc=4)
        plt.grid()
        plt.xlabel('bitrate (kbps)')
        plt.ylabel('VMAF')
        plt.title('MCL-JVC Dataset')
        plt.tight_layout()
        if not save_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(save_root, 'rd_curve_vmaf.png'))
        plt.clf()
        print('BD-RATE (VMAF): ', BD_RATE(bpp1, vmaf1, bpp2, vmaf2))


if __name__ == '__main__':
    plot_rd_curve(save_fig=True)
