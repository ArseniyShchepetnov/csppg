'''
Created on 28-02-2018.

@author: arseniy
'''

import numpy as np

def gen_random_sensing_matrix(m, n):
    '''
    Generate sensing matrix with random samples for signal of length n with m samples
    
    :param m: Number of random samples
    :param n: Signal length

    :returns: Sensing matrix, samples
    '''
    
    M = np.zeros(shape = (m, n))
    
    if n < m:
        raise Exception('Error in sensing::gen_random_sensing_matrix: n < m')
    
    samples = []    
    while len(samples) < m:
        x = np.random.randint(0, n)
        if x not in samples:
            samples.append(x)
    samples = np.sort(samples)
    
    for i in range(0, m):
        M[i, samples[i]] = 1
        
    return M, samples