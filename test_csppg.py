'''
Created on 21-02-2018 .

@author: arseniy
'''

import tools
import transforms
from scipy import signal
import gpsr
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sensing
import scipy.signal

DATADIR = '../optics/competition_data/Training_data/'

# FS = 125
FS = 25
N_SEC = 8
N = N_SEC * FS
M = N_SEC * FS
w = 125
# S = 200
S = int(N / 4)

H, H_samples = sensing.gen_random_sensing_matrix(S, N)

print(np.shape(H), len(H_samples))

W = transforms.gen_dct_matrix(M, N) * np.sqrt(2 / N)
W_inv = np.linalg.inv(W)  
print(np.matmul(W, W_inv))

dat = tools.read_mat_ieee_training(DATADIR, 6, 2)

ppg = dat['ppg1']

ppg_filt = scipy.signal.decimate(ppg, 5)
# b, a = signal.butter(4, [0.5 / FS, 15 / FS], 'bandpass', analog = False)
# ppg_filt = signal.filtfilt(b, a, ppg)

# ppg_filt = ppg

ppg_filt_1 = ppg_filt[0*N_SEC * FS:1*N_SEC * FS]
ppg_filt_1 = ppg_filt_1 / np.std(ppg_filt_1)
ppg_resamp = np.matmul(H, ppg_filt_1) 
ppg_trans = np.matmul(W, ppg_filt_1)

A = np.matmul(H, W_inv)

ppg_t = np.matmul(H, ppg_filt_1)   
Apinv = linalg.pinv(A)  
x_min_norm = np.matmul(Apinv, ppg_t)  
tau = 0.4 * np.max(np.matmul(A.T, ppg_t))
  
x_gpsr = gpsr.gpsr_bb(x_min_norm, A, ppg_t, tau = tau, alpha0 = 1, tol = 0.0000000001, iter_max = 12)  
x_debiased = gpsr.debaising(x_gpsr, A, ppg_t, tol = 0.00000000001, fix_lev = 0.01, iter_max = 12)  
  
  
f, axes = plt.subplots(2, 2)
axes[0, 0].plot(ppg_filt_1)
axes[0, 0].plot(H_samples, ppg_resamp, '.r')
axes[0, 0].set_title('True signal ' + str(N) + ' samples. ' + str(N_SEC) + ' seconds.' +'\nSensing ' + str(S) + ' samples')

axes[0, 1].plot(ppg_trans[:250], 'g')
# axes[0, 1].plot(ppg_t[:250], 'b')
axes[0, 1].plot(x_gpsr[:250], 'r')
axes[0, 1].plot(x_debiased[:250], 'g')
axes[0, 1].set_title('Spectrum')
# axes[0, 1].plot(np.matmul(W, x_gpsr)[:250], 'r')


# axes[1, 1].plot(np.matmul(W, x_gpsr), 'b')
# axes[1, 1].plot(np.matmul(W_inv, x_min_norm), 'b')
ppg_min_norm = np.matmul(W_inv, x_min_norm)
ppg_gpsr = np.matmul(W_inv, x_gpsr)
ppg_debiased = np.matmul(W_inv, x_debiased)

axes[1, 1].plot(ppg_filt_1, '--r')
axes[1, 1].plot(ppg_debiased, 'b')
axes[1, 1].set_title('GPSR + debiasing solution')


axes[1, 0].plot(ppg_filt_1, '--r')
axes[1, 0].plot(ppg_gpsr, 'b')
axes[1, 0].set_title('Reconstructed from ' + str(S) + ' samples')


print('MSE:', np.std(ppg_filt_1 - ppg_min_norm), np.std(ppg_filt_1 - ppg_debiased), np.std(ppg_filt_1 - ppg_gpsr))


plt.show()     