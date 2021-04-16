#!/usr/bin/env python3
"""contains the initialize function"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
     initializes cluster centroids for K-means
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: positive integer containing the number of clusters
    :return: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
            containing the priors for each cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster,
            initialized with K-means
        S is a numpy.ndarray of shape (k, d, d)
            containing the covariance matrices for each cluster,
            initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None

    n, d = X.shape
    pi = np.tile(1/k, (k,))
    m, _ = kmeans(X, k)
    S_ = np.identity(d)
    S = np.tile(S_, (k, 1, 1))

    return pi, m, S
