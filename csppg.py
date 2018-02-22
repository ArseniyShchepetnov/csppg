'''
Created on 21 02 2018.

@author: arseniy
'''

import numpy as np


def gpsr_bb(x0, A, W, y, tau, alpha0 = 1, alpha_lims = (1e-30, 1e+30), tol = 1e-6, iter_max = 20):
    '''
    GPSR-BB algorithm
    
    :param x0: Initial solution
    :param A: Transform matrix
    :param y: Measurement vector
    :param tau: l1-norm weight
    :param alpha0: Initial step-size parameter
    :param alpha_lims: Limits of the gradient step-size
    :param tol: Convergence tolerance
    :param iter_max: Maximum number of iterations
    
    :returns: Reconstructed solution vector
    '''
    
    # Initialization
    
    N = len(x0) 
    N2 = 2 * N 

    Aty = np.matmul(A.T, y)
    AtA = np.matmul(A.T, A)

    c = np.zeros(shape = (N2, ))
#     f = np.zeros(shape = (N2, ))
    z_k = np.zeros(shape = (N2, ))
    Bz_k = np.zeros(shape = (N2, ))
    gradF_k = np.zeros(shape = (N2, ))
    delta_k = np.zeros(shape = (N2, ))
    Bdelta_k = np.zeros(shape = (N2, ))
    z_min_k = np.zeros(shape = (N2, ))

    Z_k = np.zeros(shape = (N2, ))

    c[:N] = - Aty
    c[N:] = Aty
    
    # z_k
    z_k[:N] = x0    
    z_k[N:] = -x0
    z_k[z_k < 0] = 0
  
    # Bz
    Bz_k[:N] = np.matmul(AtA, z_k[:N] - z_k[N:])
    Bz_k[N:] = -Bz_k[:N]

    # gradient(F)
#     Wz_k = 
    
    gradF_k[:N] = c[:N] + tau * np.abs(np.matmul(W, z_k[:N])) + Bz_k[:N]
    gradF_k[N:] = c[N:] + tau * np.abs(np.matmul(W, z_k[N:])) + Bz_k[N:]

    alpha_k = alpha0
    
    k = 0
    while k < iter_max:
        
        print('\nk = ', k)

        delta_k[:] = z_k - alpha_k * gradF_k
        delta_k[delta_k < 0] = 0
        delta_k[:] -= z_k    
        
        Bdelta_k[:N] = np.matmul(AtA, delta_k[:N] - delta_k[N:])
        Bdelta_k[N:] = - Bdelta_k[:N]
        
        gamma_k = np.dot(delta_k, Bdelta_k)
        
        lambda_k = 1
        
        if gamma_k != 0:
            lambda_k = - np.dot(delta_k, gradF_k) / gamma_k  
            if lambda_k < 0:
                lambda_k = 0
            if lambda_k > 1:
                lambda_k = 1
        else:
            lambda_k = 1
            

        z_k[:] = z_k + lambda_k * delta_k
        
        if gamma_k == 0:
            alpha_k = alpha_lims[1]
        else:
            alpha_k = np.dot(delta_k, delta_k) / gamma_k

            if alpha_k < alpha_lims[0]:
                alpha_k = alpha_lims[0]
            if alpha_k > alpha_lims[1]:
                alpha_k = alpha_lims[1]

        print('lambda_k = ', lambda_k, 'gamma_k', gamma_k, 'alpha_k = ', alpha_k)

        Bz_k[:N] = np.matmul(AtA, z_k[:N] - z_k[N:])
        Bz_k[N:] = -Bz_k[:N]

        

        gradF_k[:N] = c[N:] + tau * np.abs(np.matmul(W, z_k[:N])) + Bz_k[N:]
        gradF_k[N:] = c[N:] - tau * np.abs(np.matmul(W, z_k[N:])) + Bz_k[N:]

        z_min_k[:] = np.minimum(z_k, gradF_k)
                
        tol_k = np.dot(z_min_k, z_min_k) 
        
        print('tolerance:', tol_k, tol)
        if tol_k <= tol:
            break
        
        
        k += 1
        
        
    x_gp = z_k[:N] - z_k[N:]        
        
    return x_gp    