#!/usr/bin/env python3
"""contains the variance function"""

import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :param C: numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
    :return: var, or None on failure
    """
    try:
        k, d = C.shape

        # squared euclidean distance (SED)
        # (a - b)² = a^2 - 2ab + b^2 expansion
        # (a - b)² = ||a||² + ||b||² - 2ab
        a2 = np.sum(C ** 2, axis=1)[:, np.newaxis]
        b2 = np.sum(X ** 2, axis=1)
        ab = np.matmul(C, X.T)
        SED = a2 - 2 * ab + b2

        var = np.sum(np.amin(SED, axis=0))

        return var

    except Exception:
        return None
