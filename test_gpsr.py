'''
Created on 2018-01-27

@author: arseniy

Test for GPSR. 
Signal of length N contains T random peaks with values -1 or 1. Signal is measured of M << N samples and reconstructed by GPSR.  
 
'''

import gpsr
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Signal size
N = 4096
# Sample size
M = 1024
# Number of peaks
T = 160

# Init signal 
x = np.zeros(shape = (N, ))
q = np.random.randint(0, N, size = T)
x[q] = np.sign(np.random.randn(T) - 0.5)

A0 = np.random.randn(M, N)
A = linalg.orth(A0.T).T

y = np.matmul(A, x)

Apinv = linalg.pinv(A)
x_min_norm = np.matmul(Apinv, y) 

tau = 0.1 * np.max(np.matmul(A.T, y))

# x_gpsr = gpsr.gpsr(x_min_norm + 0.01 * (np.random.randn(N)), A, y, tau = tau, beta = 0.5, mu = 0.1, alpha_lims = (1e-30, 1e+30), tol = 0.01, iter_max = 20)
x_gpsr = gpsr.gpsr_bb(x_min_norm + 0.01 * (np.random.randn(N)), A, y, tau = tau, alpha0 = 5, alpha_lims = (1e-30, 1e+30), tol = 0.0001)

x_debiased = gpsr.debaising(x_gpsr, A, y, tol = 0.01, fix_lev = 0.1, iter_max = 12)

print(tau)

f, axes = plt.subplots(2, 2)
axes[0, 0].plot(x)
axes[0, 0].set_title('True signal')

axes[0, 1].plot(x_debiased, '-r')
axes[0, 1].set_title('Reconstructed after debiasing')

axes[1, 0].plot(x_gpsr, '-b')
axes[1, 0].set_title('Reconstructed after GPSR')

axes[1, 1].plot(x - x_gpsr, '.b')
axes[1, 1].plot(x - x_debiased, '.r')
axes[1, 1].set_title('Difference from true signal')

print('MSE:', np.dot(x - x_debiased, x - x_debiased) / N, np.dot(x - x_gpsr, x - x_gpsr) / N )

plt.show()     
