#!/usr/bin/env python3
"""contains the agglomerative function"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param dist: maximum cophenetic distance for all clusters
    :return: clss, a numpy.ndarray of shape (n,)
        containing the cluster indices for each data point
    """

    Z = scipy.cluster.hierarchy.linkage(X,
                                        method='ward')

    fig = plt.figure(figsize=(25, 10))
    dn = scipy.cluster.hierarchy.dendrogram(Z,
                                            color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z,
                                            t=dist,
                                            criterion='distance')

    return clss
