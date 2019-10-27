"""
Created on 25-02-2018 .

Basis transformations.
"""
import numpy as np


def gabor_func(i_basis: int,
               i_sig: int,
               width: int,
               sig_len: int) -> float:
    """
    Gabor function.

    Parameters
    ----------
    i_basis : int
        Basis index
    i_sig : int
        Signal index
    width : int
        Gabor kernel width
    sig_len : int
        Signal length

    Returns
    -------
    float
        Gabor function value
    """
    cos_part = np.cos(2 * np.pi * i_basis * i_sig / (2 * sig_len))
    exp_part = np.exp(- i_basis ** 2
                      * (i_sig - sig_len / 2) ** 2
                      / (width * sig_len ** 2))
    return cos_part * exp_part


def gen_gabor_matrix(basis_len: int, sig_len: int, width: int) -> np.ndarray:
    """
    Gabor transformation matrix

    Parameters
    ----------
    basis_len : int
        Number of basis functions
    sig_len : int
        Number of signal samples
    width : int
        Gabor kernel width

    Returns
    -------
    np.ndarray
        Gabor transformation matrix
    """
    result = np.fromfunction(lambda i_basis, i_sig:
                             gabor_func(i_basis, i_sig, width, basis_len),
                             (basis_len, sig_len))
    return result


def transform_dct(basis_len: int, sig_len: int) -> np.ndarray:
    """
    Cosine transform

    Parameters
    ----------
    basis_len : int
        Number of basis functions
    sig_len : int
        Number of samples in signal

    Returns
    -------
    np.ndarray
        Cosine transform matrix
    """
    result = np.fromfunction(lambda i_basis, i_sig:
                             np.cos(np.pi * i_basis * (i_sig + 0.5) / sig_len),
                             (basis_len, sig_len))

    return result
