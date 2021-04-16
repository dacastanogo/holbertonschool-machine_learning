#!/usr/bin/env python3
"""contains the initialize function"""

import numpy as np


def initialize(X, k):
    """
     initializes cluster centroids for K-means
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: positive integer containing the number of clusters
    :return: numpy.ndarray of shape (k, d)
        containing the initialized centroids for each cluster,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape

    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)

    centroids = np.random.uniform(min_val, max_val, (k, d))

    return centroids
