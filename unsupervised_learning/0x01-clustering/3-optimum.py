#!/usr/bin/env python3
"""contains the optimum_k function"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance
    :param X: numpy.ndarray of shape (n, d)
        containing the data set
    :param kmin: positive integer containing the minimum
        number of clusters to check for (inclusive)
    :param kmax:  positive integer containing the maximum
        number of clusters to check for (inclusive)
    :param iterations: positive integer containing the maximum
        number of iterations for K-means
    :return: results, d_vars, or None, None on failure
        results is a list containing the outputs
            of K-means for each cluster size
        d_vars is a list containing the difference
            in variance from the smallest cluster size for each cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None
    if type(kmax) != int or kmax <= 0 or X.shape[0] < kmax:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        if k == kmin:
            initial_var = variance(X, C)

        var = variance(X, C)
        d_vars.append(initial_var - var)

    return results, d_vars
