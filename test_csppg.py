'''
Created on 21 02 2018 .

@author: arseniy
'''

import tools
import transforms
from scipy import signal
import gpsr
import csppg
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

DATADIR = '../optics/competition_data/Training_data/'

FS = 125
N_SEC = 8
N = N_SEC * FS
M = N_SEC * FS
w = 125

S = 400

H_arr = np.sort(np.unique(np.random.randint(0, N, size = S)))
print(H_arr, len(H_arr))
S = len(H_arr)
H = np.zeros(shape = (S, N))


for i in range(0, S):
    H[i, H_arr[i]] = 1



MGab = transforms.gen_dct_matrix(M, N)
MGab_inv = np.sqrt(2 / N) * np.linalg.inv(MGab)
print(np.matmul(MGab, MGab_inv))

dat = tools.read_mat_ieee_training(DATADIR, 6, 2)

ppg = dat['ppg1']
b, a = signal.butter(4, [0.5 / FS, 15 / FS], 'bandpass', analog = False)
ppg_filt = signal.filtfilt(b, a, ppg)

ppg_filt_1 = ppg_filt[:N_SEC * FS]
ppg_resamp = np.matmul(H, ppg_filt_1) 
ppg_trans = np.matmul(MGab, ppg_filt_1)

A = np.matmul(H, MGab_inv)

ppg_t = np.matmul(H, ppg_filt_1)   
Apinv = linalg.pinv(A)  
x_min_norm = np.matmul(Apinv, ppg_t)  
tau = 0.1 * np.max(np.matmul(A.T, ppg_t))
  
x_gpsr = gpsr.gpsr_bb(x_min_norm, A, ppg_t, tau = tau, alpha0 = 1, tol = 0.00000000000001)  
  
f, axes = plt.subplots(2, 2)
axes[0, 0].plot(ppg_filt_1)
axes[0, 0].set_title('True signal')

axes[1, 0].plot(ppg_filt_1, '.r')
axes[1, 0].plot(H_arr, ppg_resamp, '.b')

axes[0, 1].plot(ppg_trans[:250], 'g')
# axes[0, 1].plot(ppg_t[:250], 'b')
axes[0, 1].plot(x_gpsr[:250], 'r')
# axes[0, 1].plot(np.matmul(MGab, x_gpsr)[:250], 'r')


# axes[1, 1].plot(np.matmul(MGab, x_gpsr), 'b')
# axes[1, 1].plot(np.matmul(MGab_inv, x_min_norm), 'b')
axes[1, 1].plot(np.matmul(MGab_inv, x_gpsr), 'r')



plt.show()     