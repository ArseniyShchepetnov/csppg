'''
Created on 21 02 2018 .

@author: arseniy
'''
import scipy.io 
import numpy as np

def read_mat_ieee_training(ddir, n, t):
    '''
    '''
    str_n = str(n) if n >= 10 else '0' + str(n)
    f1 = ddir + 'DATA_' + str_n + '_TYPE0' + str(t) + '.mat'
    f2 = ddir + 'DATA_' + str_n + '_TYPE0' + str(t) + '_BPMtrace.mat'
    
    print('LOAD:', f1, f2)
    
    sig = scipy.io.loadmat(f1)
    bpm = scipy.io.loadmat(f2)
    
    return {'ecg': sig['sig'][0],
         'ppg1': sig['sig'][1],
         'ppg2': sig['sig'][2],
         'ax': sig['sig'][3],
         'ay': sig['sig'][4],
         'az': sig['sig'][5],
         'bpm': np.array([x[0] for x in bpm['BPM0']])}
    
