"""
Created on 2018-01-27

Test for GPSR.
Signal of length N contains T random peaks with values -1 or 1.
Signal is measured of M << N samples and reconstructed by GPSR.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import gpsr

SIG_LEN = 4096  # Signal size
MEASURED_LEN = 1024  # Sample size
NUM_NOT_ZEROS = 160  # Number of peaks


def main():
    """Main function"""

    # Init signal
    sig = np.zeros(shape=(SIG_LEN, ))
    pos_not_zeros = np.random.randint(0, SIG_LEN, size=NUM_NOT_ZEROS)
    sig[pos_not_zeros] = np.sign(np.random.randn(NUM_NOT_ZEROS) - 0.5)

    sensing_matrix = np.random.randn(MEASURED_LEN, SIG_LEN)
    sensing_matrix = linalg.orth(sensing_matrix.T)  # pylint: disable=no-member
    sensing_matrix = sensing_matrix.T

    measured_sig = np.matmul(sensing_matrix, sig)

    sensing_matrix_pinv = linalg.pinv(sensing_matrix)
    x_min_norm = np.matmul(sensing_matrix_pinv, measured_sig)

    tau = 0.1 * np.max(np.matmul(sensing_matrix.T, measured_sig))

    initial_solution = x_min_norm + 0.01 * (np.random.randn(SIG_LEN))

    x_gpsr = gpsr.gpsr_bb(initial_solution,
                          sensing_matrix,
                          measured_sig,
                          tau=tau, alpha0=5,
                          alpha_lims=(1e-30, 1e+30), tolerance=0.0001)

    x_debiased = gpsr.debaising(x_gpsr, sensing_matrix, measured_sig,
                                tol=0.01, fix_lev=0.1, iter_max=12)

    print(tau)

    _, axes = plt.subplots(2, 2)
    axes[0, 0].plot(sig)
    axes[0, 0].set_title('True signal')

    axes[0, 1].plot(x_debiased, '-r')
    axes[0, 1].set_title('Reconstructed after debiasing')

    axes[1, 0].plot(x_gpsr, '-b')
    axes[1, 0].set_title('Reconstructed after GPSR')

    axes[1, 1].plot(sig - x_gpsr, '.b')
    axes[1, 1].plot(sig - x_debiased, '.r')
    axes[1, 1].set_title('Difference from true signal')

    print('MSE:', np.dot(sig - x_debiased, sig - x_debiased) /
          SIG_LEN, np.dot(sig - x_gpsr, sig - x_gpsr) / SIG_LEN)

    plt.show()


if __name__ == "__main__":
    main()
