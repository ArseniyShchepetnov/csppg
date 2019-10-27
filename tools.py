"""
Created on 21-02-2018.

Tools to load data.
"""
import scipy.io
import numpy as np


def read_mat_ieee_training(data_dir, num_1, num_2):
    """
    'IEEE Signal Processing Cup 2015
    https://sites.google.com/site/researchbyzhang/ieeespcup2015'
    """

    str_n = str(num_1) if num_1 >= 10 else '0{}'.format(num_1)

    file_1 = '{}DATA_{}_TYPE0{}.mat'.format(data_dir, str_n, num_2)
    file_2 = '{}DATA_{}_TYPE0{}_BPMtrace.mat'.format(data_dir, str_n, num_2)

    print('LOAD:', file_1, file_2)

    sig = scipy.io.loadmat(file_1)
    bpm = scipy.io.loadmat(file_2)

    data_dict = {'ecg': sig['sig'][0],
                 'ppg1': sig['sig'][1],
                 'ppg2': sig['sig'][2],
                 'ax': sig['sig'][3],
                 'ay': sig['sig'][4],
                 'az': sig['sig'][5],
                 'bpm': np.array([x[0] for x in bpm['BPM0']])}
    return data_dict
