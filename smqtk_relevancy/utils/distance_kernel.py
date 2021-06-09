"""
LICENCE
-------
Copyright 2013-2020 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

import numpy as np


def compute_distance_matrix(m1: np.ndarray, m2: np.ndarray,
                            dist_func: np.ndarray, row_wise: bool = False) \
                            -> np.ndarray:
    """
    Function for computing the pair-wise distance matrix between two arrays of
    vectors. Both matrices must have the same number of columns.
    """
    if m1.ndim == 1:
        m1 = m1[np.newaxis]
    if m2.ndim == 1:
        m2 = m2[np.newaxis]
    k = np.ndarray((m1.shape[0], m2.shape[0]), dtype=float)
    if row_wise:
        # row wise
        for i in range(m1.shape[0]):
            k[i, :] = dist_func(m1[i], m2)
    else:
        for i in range(m1.shape[0]):
            for j in range(m2.shape[0]):
                k[i, j] = dist_func(m1[i], m2[j])
    return k
