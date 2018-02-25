'''
Created on 25-02-2018 .

@author: arseniy
'''

import tools
import numpy as np
import transforms
import matplotlib.pyplot as plt
import scipy.signal as signal

FS = 125

DATADIR = '../optics/competition_data/Training_data/'
dat = tools.read_mat_ieee_training(DATADIR, 6, 2)
ppg = dat['ppg1']
b, a = signal.butter(4, [0.5 / FS, 15 / FS], 'bandpass', analog = False)
ppg_filt = signal.filtfilt(b, a, ppg)

N = 1000
ppg_filt_1 = ppg_filt[:N]

Mdct = transforms.gen_dct_matrix(N, N)
Mdct_inv = np.linalg.inv(Mdct) 
print(np.matmul(Mdct, Mdct_inv))

ppg_filt_1_dct = np.matmul(Mdct, ppg_filt_1)
ppg_filt_1_gab = np.matmul(transforms.gen_gabor_matrix(N, N, 500), ppg_filt_1)

f, axes = plt.subplots(2, 2)
axes[0, 0].plot(ppg_filt_1)

axes[0, 1].plot(ppg_filt_1_dct[:150])

axes[1, 0].plot(ppg_filt_1_gab[:150])

axes[1, 1].plot(np.matmul(Mdct_inv, ppg_filt_1_dct))

plt.show()     