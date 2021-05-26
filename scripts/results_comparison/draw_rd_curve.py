import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from basicsr.utils import mkdir


if __name__ == '__main__':

    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    LineWidth = 2

    bpp = [0.00053461, 0.00036419, 0.00025088, 0.00017527]
    psnr = [31.5574, 31.0808, 30.2522, 28.9882]
    msssim = []
    sr_baseline, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='SR Baseline')

    bpp = [0.00062365, 0.00040931, 0.00027154, 0.00018400]
    psnr = [32.0263, 31.6678, 30.8174, 29.3943]
    msssim = []
    sr_cp_crf23, = plt.plot(bpp, psnr, "c-*", linewidth=LineWidth, label='SR+CP CRF23')

    bpp = [0.00068612, 0.00044797, 0.00029307, 0.00019387]
    psnr = [31.5432, 31.3855, 30.7653, 29.4650]
    msssim = []
    sr_cp_crf28, = plt.plot(bpp, psnr, "r-*", linewidth=LineWidth, label='SR+CP CRF28')

    save_path = 'vimeo90k.png'

    plt.legend(handles=[sr_baseline, sr_cp_crf23, sr_cp_crf28], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR')
    plt.title('Vimeo90k Dataset')
    # plt.savefig(save_path)
    # plt.clf()
    plt.show()
