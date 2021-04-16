#!/usr/bin/env python3
"""contains the kmeans function"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm


def static_plot(X, C, clss, i):
    """
    :param array:
    :param colors:
    :return:
    """
    ndigits = 3  # to determine filename padding
    outdir = 'frames'

    # create directory if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.ioff()
    with plt.xkcd():
        fig = plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=2, c=clss, cmap='Set1')
        plt.scatter(C[:, 0], C[:, 1], s=100, marker='X',
                    c=list(range(C.shape[0])), cmap='Set1')

        # save as png
        outfile = os.path.join(outdir, "frame-" + str(i).
                               zfill(ndigits) + ".png")
        fig.savefig(outfile)
    plt.ion()


def kmeans(X, k, iterations=1000):
    """
     initializes cluster centroids for K-means
    :param X: numpy.ndarray of shape (n, d)
        containing the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: positive integer containing the number of clusters
    :param iterations: positive integer containing the maximum number
        of iterations that should be performed
    :return: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,)
            containing the index of the cluster in C that
            each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    # initialization
    n, d = X.shape

    clss = np.zeros(n)

    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)

    C = np.random.uniform(min_val, max_val, (k, d))
    C_prev = np.copy(C)

    # generate intermediate plots
    static_plot(X, C, clss, 0)

    X_ = X[:, :, np.newaxis]
    C_ = C.T[np.newaxis, :, :]
    diff = X_ - C_
    D = np.linalg.norm(diff, axis=1)

    clss = np.argmin(D, axis=1)

    for i in range(iterations):

        # generate intermediate plots
        static_plot(X, C, clss, i + 1)

        for j in range(k):
            # recalculate centroids
            index = np.where(clss == j)
            if len(index[0]) == 0:
                C[j] = np.random.uniform(min_val, max_val, (1, d))
            else:
                C[j] = np.mean(X[index], axis=0)

        X_ = X[:, :, np.newaxis]
        C_ = C.T[np.newaxis, :, :]
        diff = X_ - C_
        D = np.linalg.norm(diff, axis=1)

        clss = np.argmin(D, axis=1)

        if (C == C_prev).all():
            return C, clss
        C_prev = np.copy(C)

    print(i)
    return C, clss
