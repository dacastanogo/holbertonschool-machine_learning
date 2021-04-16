#!/usr/bin/env python3
"""contains the kmeans function"""

import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param k: number of clusters
    :return: C, clss
        C is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,)
            containing the index of the cluster in C that
            each data point belongs to
    """
    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)
    C = Kmean.cluster_centers_
    clss = Kmean.labels_

    return C, clss
