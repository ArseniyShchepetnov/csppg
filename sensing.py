"""
Created on 28-02-2018.

@author: arseniy
"""
from typing import Tuple

import numpy as np


def gen_random_sensing_matrix(n_samples: int,
                              sig_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sensing matrix with random samples for signal of
    length `sig_len` with `n_samples` samples.

    Parameters
    ----------
    n_samples : int
        Number of random samples
    sig_len : int
        Signal length

    Returns
    -------
    np.ndarray, np.ndarray
        Sensing matrix, samples

    Raises
    ------
    ValueError
        [description]
    """

    sensing_matrix = np.zeros(shape=(n_samples, sig_len))

    if sig_len < n_samples:
        raise ValueError(
            'Error in sensing::gen_random_sensing_matrix: sig_len < n_samples')

    samples = []
    while len(samples) < n_samples:

        pos = np.random.randint(0, sig_len)
        if pos not in samples:
            samples.append(pos)

    samples = np.sort(samples)

    for i_sample in range(0, n_samples):
        sensing_matrix[i_sample, samples[i_sample]] = 1

    return sensing_matrix, samples
