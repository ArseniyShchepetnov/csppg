"""
Created on 2018-01-14

`Gradient Projection for Sparse Reconstruction: Application to
Compressed Sensing and Other Inverse Problems
http://ieeexplore.ieee.org/document/4407762/`_
"""
import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def gpsr(initial_solution: Union[np.ndarray, list],  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
         transform: np.ndarray,
         signal: Union[np.ndarray, list],
         tau: float,
         sb_beta: Optional[float] = 0.5,
         sb_mu: Optional[float] = 0.1,
         alpha_lims: Optional[Tuple[float, float]] = None,
         tolerance: float = 1e-6,
         iter_max: int = 20) -> np.ndarray:
    """
    Basic Gradient Projection for Sparse Reconstruction

    Parameters
    ----------
    initial_solution : Union[np.ndarray, list]
        Initial solution
    transform : np.ndarray
        Transform matrix
    signal : Union[np.ndarray, list]
        Measurement vector
    tau : float
        l1-norm weight
    beta : float, optional
        Search-back parameter, by default 0.5
    sb_mu : float, optional
        Search-back stop condition parameter, by default 0.1
    alpha_lims : Tuple[float, float], optional
        Limits of the gradient step-size, by default (0.01, 0.9)
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
    iter_max : int, optional
        Maximum number of iterations, by default 20

    Returns
    -------
    np.ndarray
        Reconstructed solution vector
    """

    if alpha_lims is None:
        alpha_lims = (1e-30, 1e+30)

    logger.info('GPSR-Basic')

    sig_len = len(initial_solution)
    sig_len_2 = sig_len * 2

    trans_t_sig = np.matmul(transform.T, signal)
    trans_t_trans = np.matmul(transform.T, transform)

    const = np.zeros(shape=(sig_len_2, ))
    z_k = np.zeros(shape=(sig_len_2, ))
    bz_k = np.zeros(shape=(sig_len_2, ))
    bg_k = np.zeros(shape=(sig_len_2, ))
    bw_k = np.zeros(shape=(sig_len_2, ))
    grad_f_k = np.zeros(shape=(sig_len_2, ))
    g_k = np.zeros(shape=(sig_len_2, ))
    w_k = np.zeros(shape=(sig_len_2, ))
    d_k = np.zeros(shape=(sig_len_2, ))

    # const initialization
    const[:sig_len] = np.full((sig_len, ), tau) - trans_t_sig
    const[sig_len:] = np.full((sig_len, ), tau) + trans_t_sig

    # z_k initialization
    z_k[:sig_len] = initial_solution
    z_k[sig_len:] = -initial_solution
    z_k[z_k < 0] = 0

    bz_k[:sig_len] = np.matmul(trans_t_trans, z_k[:sig_len] - z_k[sig_len:])
    bz_k[sig_len:] = - bz_k[:sig_len]

    grad_f_k[:] = const + bz_k

    for k_iter in range(iter_max):

        g_k = np.where(np.any([grad_f_k < 0, z_k > 0], axis=0), grad_f_k, 0)

        # compute alpha0
        bg_k[:sig_len] = np.matmul(trans_t_trans,
                                   g_k[:sig_len] - g_k[sig_len:])

        bg_k[sig_len:] = - bg_k[:sig_len]
        alpha0 = np.dot(g_k, g_k) / np.dot(g_k, bg_k)

        # backtracking search
        alpha_k = alpha0

        if alpha_k > alpha_lims[1]:
            alpha_k = alpha_lims[1]

        if alpha_k < alpha_lims[0]:
            alpha_k = alpha_lims[0]

        fz_k = np.dot(const, z_k) + 0.5 * np.dot(z_k, bz_k)
        while True:

            w_k[:] = z_k - alpha_k * grad_f_k
            w_k[w_k < 0] = 0

            d_k[:] = z_k - w_k

            bw_k[:sig_len] = np.matmul(trans_t_trans,
                                       w_k[:sig_len] - w_k[sig_len:])
            bw_k[sig_len:] = - bw_k[:sig_len]

            fw_k = np.dot(const, w_k) + 0.5 * np.dot(w_k, bw_k)

            fw_k_1 = fz_k - sb_mu * np.vdot(grad_f_k, d_k)

            if fw_k <= fw_k_1:
                break

            alpha_k *= sb_beta

        # end of iteration

        tol_k = np.dot(d_k, d_k)

        logger.info('k_iter = %d, tol_k = %f / %f', k_iter, tol_k, tolerance)
        if tol_k <= tolerance:
            break

        z_k[:] = w_k

        bz_k[:sig_len] = np.matmul(trans_t_trans,
                                   z_k[:sig_len] - z_k[sig_len:])
        bz_k[sig_len:] = - bz_k[:sig_len]

        grad_f_k[:] = const + bz_k

    result = z_k[:sig_len] - z_k[sig_len:]

    return result


def gpsr_bb(initial_solution: Union[np.ndarray, list],  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
            transform: np.ndarray,
            signal: Union[np.ndarray, list],
            tau: float,
            alpha0: float = 1,
            alpha_lims: Optional[Tuple[float, float]] = None,
            tolerance: float = 1e-6,
            iter_max: int = 20) -> np.ndarray:
    """
    GPSR-BB algorithm

    Parameters
    ----------
    initial_solution : Union[np.ndarray, list]
        Initial solution
    transform : np.ndarray
        Transform matrix
    signal : Union[np.ndarray, list]
        Measurement vector
    tau : float
        l1-norm weight
    alpha0 : float, optional
        Initial step-size parameter, by default 1
    alpha_lims : Tuple[float, float], optional
        Limits of the gradient step-size, by default None
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
    iter_max : int, optional
        Maximum number of iterations, by default 20

    Returns
    -------
    np.ndarray
        Reconstructed solution vector
    """

    logger.info('GPSR-BB')

    if alpha_lims is None:
        alpha_lims = (1e-30, 1e+30)

    # Initialization

    sig_len = len(initial_solution)
    sig_len_2 = 2 * sig_len

    trans_t_sig = np.matmul(transform.T, signal)
    trans_t_trans = np.matmul(transform.T, transform)

    const = np.zeros(shape=(sig_len_2, ))
    z_k = np.zeros(shape=(sig_len_2, ))
    b_z_k = np.zeros(shape=(sig_len_2, ))
    grad_f_k = np.zeros(shape=(sig_len_2, ))
    delta_k = np.zeros(shape=(sig_len_2, ))
    b_delta_k = np.zeros(shape=(sig_len_2, ))
    z_min_k = np.zeros(shape=(sig_len_2, ))

    const[:sig_len] = np.full((sig_len, ), tau) - trans_t_sig
    const[sig_len:] = np.full((sig_len, ), tau) + trans_t_sig

    # z_k
    z_k[:sig_len] = initial_solution
    z_k[sig_len:] = -initial_solution
    z_k[z_k < 0] = 0

    # Bz
    b_z_k[:sig_len] = np.matmul(trans_t_trans,
                                z_k[:sig_len] - z_k[sig_len:])
    b_z_k[sig_len:] = -b_z_k[:sig_len]

    # gradient(F)
    grad_f_k[:] = const + b_z_k

    init_sol_norm = np.dot(initial_solution, initial_solution)

    alpha_k = alpha0

    for k_iter in range(iter_max):

        delta_k[:] = z_k - alpha_k * grad_f_k
        delta_k[delta_k < 0] = 0
        delta_k[:] -= z_k

        b_delta_k[:sig_len] = np.matmul(trans_t_trans,
                                        delta_k[:sig_len] - delta_k[sig_len:])
        b_delta_k[sig_len:] = - b_delta_k[:sig_len]

        gamma_k = np.dot(delta_k, b_delta_k)

        lambda_k = 1

        if gamma_k != 0:
            lambda_k = - np.dot(delta_k, grad_f_k) / gamma_k
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

        b_z_k[:sig_len] = np.matmul(trans_t_trans,
                                    z_k[:sig_len] - z_k[sig_len:])
        b_z_k[sig_len:] = -b_z_k[:sig_len]

        grad_f_k[:] = const + b_z_k

        z_min_k[:] = np.minimum(z_k, grad_f_k)

        tol_k = np.dot(z_min_k, z_min_k) / init_sol_norm

        logger.info('k_iter = %d, tol_k = %f / %f', k_iter, tol_k, tolerance)
        if tol_k <= tolerance:
            break

    x_gp = z_k[:sig_len] - z_k[sig_len:]

    return x_gp


def debaising(initial_solution: Union[np.ndarray, list],  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
              transform: np.ndarray,
              signal: Union[np.ndarray, list],
              tol: float = 0.1,
              iter_max: float = 10,
              fix_lev: float = 0.1) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    initial_solution : Union[np.ndarray, list]
        Initial solution
    transform : np.ndarray
        Transform matrix
    signal : Union[np.ndarray, list]
        Measurement vector
    tol : float, optional
        Convergence tolerance, by default 0.1
    iter_max : float, optional
        Maximum number of iterations, by default 10
    fix_lev : float, optional
        Fixing threshold for sparsity, by default 0.1

    Returns
    -------
    np.ndarray
        Debiased solution
    """

    logger.info('Debiasing')

    x_k = initial_solution
    trans_t_trans = np.matmul(transform.T, transform)
    trans_t_sig = np.matmul(transform.T, signal)

    f_0 = np.dot(signal - np.matmul(transform, initial_solution),
                 signal - np.matmul(transform, initial_solution))

    r_k = trans_t_sig - np.matmul(trans_t_trans, x_k)
    p_k = np.array(r_k)

    k_iter = 0
    while True:

        alpha_k = np.dot(r_k, r_k) / np.dot(p_k, np.matmul(trans_t_trans, p_k))

        p_k[np.abs(initial_solution) < fix_lev] = 0

        x_k = x_k + alpha_k * p_k
        r_k1 = r_k - alpha_k * np.matmul(trans_t_trans, p_k)

        f_k = np.dot(r_k1, r_k1)
        tolerance = tol * f_0
        logger.info('k_iter = %d, F_k = %f / %f', k_iter, f_k, tolerance)

        if f_k <= tol * f_0 or k_iter > iter_max:
            break

        beta_k = np.dot(r_k1, r_k1) / np.dot(r_k, r_k)

        p_k = r_k1 + beta_k * p_k
        r_k = r_k1

        k_iter += 1

    return x_k
