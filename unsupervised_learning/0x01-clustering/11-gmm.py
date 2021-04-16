#!/usr/bin/env python3
"""contains the gmm function"""

import sklearn.mixture


def gmm(X, k):
    """
    calculates a GMM from a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param k: number of clusters
    :return: Returns: pi, m, S, clss, bic
        pi is a numpy.ndarray of shape (k,)
            containing the cluster priors
        m is a numpy.ndarray of shape (k, d)
            containing the centroid means
        S is a numpy.ndarray of shape (k, d, d)
            containing the covariance matrices
        clss is a numpy.ndarray of shape (n,)
            containing the cluster indices for each data point
        bic is a numpy.ndarray of shape (kmax - kmin + 1)
            containing the BIC value for each cluster size tested
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_params = GMM.fit(X)

    clss = GMM.predict(X)

    pi = gmm_params.weights_
    m = gmm_params.means_
    S = gmm_params.covariances_
    bic = GMM.bic(X)

    return pi, m, S, clss, bic
