#!/usr/bin/env python3
"""contains the expectation_maximization function"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def static_plot(X, m, g, k, i):
    """
    plot every iteration
    """
    ndigits = 3  # to determine filename padding
    outdir = 'frames2'

    # create directory if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.ioff()
    with plt.xkcd():
        fig = plt.figure()

        plt.scatter(X[:, 0], X[:, 1], s=7, c=clss, cmap='jet')
        plt.scatter(m[:, 0], m[:, 1], s=100, c=np.arange(k),
                    marker='X', cmap='jet')

        # save as png
        outfile = os.path.join(outdir, "frame-" + str(i).
                               zfill(ndigits) + ".png")
        fig.savefig(outfile)
    plt.ion()


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
        if verbose and (i % 3 == 0):
            print('Log Likelihood after {} iterations: {}'.format(i,
                                                                  l_.round(5)))
            static_plot(X, m, g, k, i)

        pi, m, S = maximization(X, g)
        g, l_ = expectation(X, pi, m, S)

        if abs(l_prev - l_) <= tol:
            break

        l_prev = l_

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(i + 1,
                                                              l_.round(5)))
        static_plot(X, m, g, k, i)

    return pi, m, S, g, l_,
