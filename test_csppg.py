'''
Created on 21 02 2018 .

@author: arseniy
'''

import tools
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

S = 500

H_arr = np.random.randint(0, N, size = S)

H = np.zeros(shape = (S, N))
for i in range(0, S):
    H[i, H_arr[i]] = 1

MGab = csppg.gen_gabor_matrix(M, N, w)

dat = tools.read_mat_ieee_training(DATADIR, 6, 2)

ppg = dat['ppg1']
b, a = signal.butter(4, [0.5 / FS, 15 / FS], 'bandpass', analog = False)
ppg_filt = signal.filtfilt(b, a, ppg)

ppg_filt_1 = ppg_filt[:N_SEC * FS]
ppg_resamp = np.matmul(H, ppg_filt_1) 
ppg_trans = np.matmul(MGab, ppg_filt_1)
  
  
f, axes = plt.subplots(2, 2)
axes[0, 0].plot(ppg_filt_1)
axes[0, 0].set_title('True signal')

axes[1, 0].plot(ppg_filt_1, '.r')
axes[1, 0].plot(H_arr, ppg_resamp, '.b')

axes[0, 1].plot(MGab[-1, :])
axes[1, 1].plot(np.matmul(np.linalg.inv(MGab), ppg_trans))



plt.show()     