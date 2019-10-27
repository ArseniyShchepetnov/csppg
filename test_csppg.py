"""
Created on 21-02-2018 .

Testing compressed sensing algorithms for PPG signals.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import linalg

import gpsr
import sensing
import tools
import transforms

DATADIR = '../optics/competition_data/Training_data/'

SAMPLING_RATE = 25
N_SEC = 8
SIG_LEN = N_SEC * SAMPLING_RATE
BASIS_LEN = N_SEC * SAMPLING_RATE
MEASURED_LEN = int(SIG_LEN / 4)


def main():  # pylint: disable=too-many-locals
    """Run test"""
    sensing_matrix_0, _ = sensing.gen_random_sensing_matrix(
        MEASURED_LEN, SIG_LEN)

    transform = transforms.transform_dct(BASIS_LEN, SIG_LEN)
    transform = transform * np.sqrt(2 / SIG_LEN)
    transform_inv = np.linalg.inv(transform)

    print(np.matmul(transform, transform_inv))

    dat = tools.read_mat_ieee_training(DATADIR, 6, 2)

    ppg = dat['ppg1']

    ppg_filt = scipy.signal.decimate(ppg, 5)

    ppg_filt_1 = ppg_filt[0*N_SEC * SAMPLING_RATE:1*N_SEC * SAMPLING_RATE]
    ppg_filt_1 = ppg_filt_1 / np.std(ppg_filt_1)
    ppg_resamp = np.matmul(sensing_matrix_0, ppg_filt_1)
    ppg_trans = np.matmul(transform, ppg_filt_1)

    sensing_matrix = np.matmul(sensing_matrix_0, transform_inv)

    ppg_t = np.matmul(sensing_matrix_0, ppg_filt_1)
    sensing_matrix_pinv = linalg.pinv(sensing_matrix)
    x_min_norm = np.matmul(sensing_matrix_pinv, ppg_t)
    tau = 0.4 * np.max(np.matmul(sensing_matrix.T, ppg_t))

    x_gpsr = gpsr.gpsr_bb(x_min_norm, sensing_matrix, ppg_t, tau=tau,
                          alpha0=1, tolerance=0.0000000001, iter_max=12)
    x_debiased = gpsr.debaising(x_gpsr,
                                sensing_matrix,
                                ppg_t,
                                tol=0.00000000001, fix_lev=0.01, iter_max=12)

    _, axes = plt.subplots(2, 2)
    axes[0, 0].plot(ppg_filt_1)
    axes[0, 0].plot(sensing_matrix_0, ppg_resamp, '.r')
    axes[0, 0].set_title('True signal ' + str(SIG_LEN) + ' samples. '
                         + str(N_SEC) + ' seconds.' + '\nSensing '
                         + str(MEASURED_LEN) + ' samples')

    axes[0, 1].plot(ppg_trans[:250], 'g')
    # axes[0, 1].plot(ppg_t[:250], 'b')
    axes[0, 1].plot(x_gpsr[:250], 'r')
    axes[0, 1].plot(x_debiased[:250], 'g')
    axes[0, 1].set_title('Spectrum')
    # axes[0, 1].plot(np.matmul(W, x_gpsr)[:250], 'r')

    # axes[1, 1].plot(np.matmul(W, x_gpsr), 'b')
    # axes[1, 1].plot(np.matmul(W_inv, x_min_norm), 'b')
    ppg_min_norm = np.matmul(transform_inv, x_min_norm)
    ppg_gpsr = np.matmul(transform_inv, x_gpsr)
    ppg_debiased = np.matmul(transform_inv, x_debiased)

    axes[1, 1].plot(ppg_filt_1, '--r')
    axes[1, 1].plot(ppg_debiased, 'b')
    axes[1, 1].set_title('GPSR + debiasing solution')

    axes[1, 0].plot(ppg_filt_1, '--r')
    axes[1, 0].plot(ppg_gpsr, 'b')
    axes[1, 0].set_title('Reconstructed from ' +
                         str(MEASURED_LEN) + ' samples')

    print('MSE:', np.std(ppg_filt_1 - ppg_min_norm),
          np.std(ppg_filt_1 - ppg_debiased), np.std(ppg_filt_1 - ppg_gpsr))

    plt.show()


if __name__ == "__main__":
    main()
