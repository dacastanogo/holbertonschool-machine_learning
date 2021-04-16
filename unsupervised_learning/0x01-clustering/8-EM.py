#!/usr/bin/env python3
"""contains the expectation_maximization function"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """

    :param X: numpy.ndarray of shape (n, d)
        containing the data set
    :param k: positive integer containing the number of clusters
    :param iterations: positive integer containing the maximum number
        of iterations for the algorithm
    :param tol: non-negative float containing tolerance of the log likelihood,
        used to determine early stopping
    :param verbose: boolean that determines if you
        should print information about the algorithm
    :return: pi, m, S, g, l, or None, None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
            containing the priors for each cluster
        m is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d)
            containing the covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n)
            containing the probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    # conditions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    # initialization
    n, d = X.shape

    l_prev = 0
    pi, m, S = initialize(X, k)
    g, l_ = expectation(X, pi, m, S)

    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print('Log Likelihood after {} iterations: {}'.format(i,
                                                                  l_.round(5)))

        pi, m, S = maximization(X, g)
        g, l_ = expectation(X, pi, m, S)

        if abs(l_prev - l_) <= tol:
            break

        l_prev = l_

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(i + 1,
                                                              l_.round(5)))

    return pi, m, S, g, l_,
