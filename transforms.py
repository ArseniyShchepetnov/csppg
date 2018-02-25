'''
Created on 25-02-2018 .

@author: arseniy
'''

import numpy as np

def gabor_func(i, j, w, n):
    '''
    Gabor basis function
    '''
    return np.cos(2 * np.pi * (i) * (j) / (2 * n)) * np.exp(- (i) ** 2 * (j - n / 2) ** 2 / (w * n ** 2))

def gen_gabor_matrix(m, n, w):
    '''
    Generate Gabor basis matrix
    '''
    return np.fromfunction(lambda i, j: gabor_func(i, j, w, m), (m, n))
 
def gen_dct_matrix(m, n):
    
    '''
    Generate DCT basis matrix
    '''
    return np.fromfunction(lambda i, j: np.cos(np.pi * (i) * (j + 0.5) / n), (m, n))    
