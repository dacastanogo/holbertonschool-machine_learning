#!/usr/bin/env python3
"""contains the maximization function"""

import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM
    :param X: numpy.ndarray of shape (n, d)
        containing the data set
    :param g: numpy.ndarray of shape (k, n)
        containing the posterior probabilities for each data point
        in each cluster
    :return: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
            containing the updated priors for each cluster
        m is a numpy.ndarray of shape (k, d)
            containing the updated centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d)
            containing the updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None
    n, d = X.shape

    k = g.shape[0]

    # sum of gi equal to 1
    probs = np.sum(g, axis=0)
    tester = np.ones((n,))
    if not np.isclose(probs, tester).all():
        return None, None, None

    # initialization
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n

        m[i] = np.matmul(g[i], X) / np.sum(g[i])

        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
