'''
Created on 25-02-2018 .

Test basis transformation.
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import tools
import transforms

SAMPLE_RATE = 125
SIG_LEN = 1000
DATADIR = '../optics/competition_data/Training_data/'


def main():
    """Main"""
    dat = tools.read_mat_ieee_training(DATADIR, 6, 2)
    ppg = dat['ppg1']
    b_coeffs, a_coeffs = signal.butter(4,
                                       [0.5 / SAMPLE_RATE, 15 / SAMPLE_RATE],
                                       'bandpass', analog=False)
    ppg_filt = signal.filtfilt(b_coeffs, a_coeffs, ppg)

    ppg_filt_1 = ppg_filt[:SIG_LEN]

    transform_dct = transforms.gen_dct_matrix(SIG_LEN, SIG_LEN)
    transform_dct_inv = np.linalg.inv(transform_dct)

    ppg_filt_1_dct = np.matmul(transform_dct, ppg_filt_1)
    gab_matrix = transforms.gen_gabor_matrix(SIG_LEN, SIG_LEN, 500)
    ppg_filt_1_gab = np.matmul(gab_matrix, ppg_filt_1)

    _, axes = plt.subplots(2, 2)
    axes[0, 0].plot(ppg_filt_1)

    axes[0, 1].plot(ppg_filt_1_dct[:150])

    axes[1, 0].plot(ppg_filt_1_gab[:150])

    axes[1, 1].plot(np.matmul(transform_dct_inv, ppg_filt_1_dct))

    plt.show()


if __name__ == "__main__":
    main()
