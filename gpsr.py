'''
Created on 2018-01-14

@author: arseniy


`Gradient Projection for Sparse Reconstruction: Application to Compressed Sensing and Other Inverse Problems
http://ieeexplore.ieee.org/document/4407762/`_
'''


import numpy as np



def gpsr(x0, A, y, tau, beta = 0.5, mu = 0.1, alpha_lims = (0.01, 0.9), tol = 1e-6, iter_max = 20):
    '''
    Basic Gradient Projection for Sparse Reconstruction
    
    :param x0: Initial solution
    :param A: Transform matrix
    :param y: Measurement vector
    :param tau: l1-norm weight
    :param beta: Search-back parameter
    :param mu: Search-back stop condition parameter
    :param alpha_lims: Limits of the gradient step-size
    :param tol: Convergence tolerance
    :param iter_max: Maximum number of iterations
    
    :returns: Reconstructed solution vector
    
    :Example:
    
    '''
    
    print('\nGPSR-Basic\n')
    
    N = len(x0)  
    N2 = N * 2

    Aty = np.matmul(A.T, y)
    AtA = np.matmul(A.T, A)

    c = np.zeros(shape = (N2, ))
    z_k = np.zeros(shape = (N2, ))
    Bz_k = np.zeros(shape = (N2, ))
    Bg_k = np.zeros(shape = (N2, ))
    Bw_k = np.zeros(shape = (N2, ))   
    gradF_k = np.zeros(shape = (N2, ))
    g_k = np.zeros(shape = (N2, )) 
    w_k = np.zeros(shape = (N2, ))
    d_k = np.zeros(shape = (N2, )) 

    # c
    c[:N] = np.full((N, ), tau) - Aty
    c[N:] = np.full((N, ), tau) + Aty
    
    # z_k
    z_k[:N] = x0
    z_k[N:] = -x0
    z_k[z_k < 0] = 0
    

    Bz_k[:N] = np.matmul(AtA, z_k[:N] - z_k[N:])
    Bz_k[N:] = - Bz_k[:N]
    
    gradF_k[:] = c + Bz_k
    
    k = 0
    while k < iter_max:
        print('\nk = ', k)

        g_k = np.where(np.any([gradF_k < 0, z_k > 0], axis = 0), gradF_k, 0)
            
        # compute alpha0            
        Bg_k[:N] = np.matmul(AtA, g_k[:N] - g_k[N:])
        Bg_k[N:] = - Bg_k[:N]    
        alpha0 = np.dot(g_k, g_k) / np.dot(g_k, Bg_k)  

        # backtracking search
        alpha_k = alpha0        

        if alpha_k > alpha_lims[1]:
            alpha_k = alpha_lims[1]

        if alpha_k < alpha_lims[0]:
            alpha_k = alpha_lims[0]

        Fz_k = np.dot(c, z_k) + 0.5 * np.dot(z_k, Bz_k)              
        while True:

            w_k[:] = z_k - alpha_k * gradF_k
            w_k[w_k < 0] = 0
           
            d_k[:] = z_k - w_k
            
            Bw_k[:N] = np.matmul(AtA, w_k[:N] - w_k[N:])
            Bw_k[N:] = - Bw_k[:N]
            
            Fw_k = np.dot(c, w_k) + 0.5 * np.dot(w_k, Bw_k)
            
            Fw_k_1 = Fz_k - mu * np.vdot(gradF_k, d_k) 
                       
            if Fw_k <= Fw_k_1:
                break
            
            alpha_k *= beta
        
        # end of iteration
        
        #x_saved.append(u_k - v_k)
        tol_k = np.dot(d_k, d_k)

        if tol_k <= tol:
            break
        
        z_k[:] = w_k

        Bz_k[:N] = np.matmul(AtA, z_k[:N] - z_k[N:])
        Bz_k[N:] = - Bz_k[:N]
        
        gradF_k[:] = c + Bz_k        

        k += 1

    return z_k[:N] - z_k[N:]
    

def gpsr_bb(x0, A, y, tau, alpha0 = 1, alpha_lims = (1e-30, 1e+30), tol = 1e-6, iter_max = 20):
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
    z_k = np.zeros(shape = (N2, ))
    Bz_k = np.zeros(shape = (N2, ))
    gradF_k = np.zeros(shape = (N2, ))
    delta_k = np.zeros(shape = (N2, ))
    Bdelta_k = np.zeros(shape = (N2, ))
    z_min_k = np.zeros(shape = (N2, ))

    c[:N] = np.full((N, ), tau) - Aty
    c[N:] = np.full((N, ), tau) + Aty
    
    # z_k

    z_k[:N] = x0    
    z_k[N:] = -x0
    z_k[z_k < 0] = 0
    
    # Bz
    Bz_k[:N] = np.matmul(AtA, z_k[:N] - z_k[N:])
    Bz_k[N:] = -Bz_k[:N]

    # gradient(F)
    gradF_k[:] = c + Bz_k

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

        gradF_k[:] = c + Bz_k

        z_min_k[:] = np.minimum(z_k, gradF_k)
                
        tol_k = np.dot(z_min_k, z_min_k) / np.dot(x0, x0)
        
        print('tolerance:', tol_k, tol)
        if tol_k <= tol:
            break
        
        
        k += 1
        
        
    x_gp = z_k[:N] - z_k[N:]        
        
    return x_gp


def debaising(x0, A, y, tol = 0.1, iter_max = 10, fix_lev = 0.1):
    '''
    Debiasing after GPSR.
    
    :param x0: Initial solution
    :param A: Transform matrix
    :param y: Measurement vector    
    :param tol: Convergence tolerance
    :param iter_max: Maximum number of iterations
    :param fix_lev: Fixing threshold for sparsity 
    
    :returns: Debiased solution
    '''
    print('\nDebiasing\n')
    
    N = len(x0)
    
    x_k = x0
    B = np.matmul(A.T, A)
    b = np.matmul(A.T, y)
    
    F0 = np.dot(y - np.matmul(A, x0), y - np.matmul(A, x0))
        
    r_k = b - np.matmul(B, x_k)
    p_k = np.array(r_k) 
    
    k = 0
    while True:
        print(k)
        alpha_k = np.dot(r_k, r_k) / np.dot(p_k, np.matmul(B, p_k))
        
        p_k[np.abs(x0) < fix_lev] = 0
        
        x_k = x_k + alpha_k * p_k 
        r_k1 = r_k - alpha_k * np.matmul(B, p_k) 
        
        F_k = np.dot(r_k1, r_k1)
        print('tolerance:', F_k, tol * F0)
        if F_k <= tol * F0 or k > iter_max:
            break
        
        beta_k = np.dot(r_k1, r_k1) / np.dot(r_k, r_k)
        
        p_k = r_k1 + beta_k * p_k 
        r_k = r_k1

        k += 1

    return x_k





