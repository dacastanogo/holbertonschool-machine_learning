#!/usr/bin/env python3
"""contains the pdf function"""

import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution
    :param X: numpy.ndarray of shape (n, d)
        containing the data points whose PDF should be evaluated
    :param m: numpy.ndarray of shape (d,)
        containing the mean of the distribution
    :param S: numpy.ndarray of shape (d, d)
        containing the covariance of the distribution
    :return: P, or None on failure
        P is a numpy.ndarray of shape (n,)
            containing the PDF values for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    Q = np.linalg.inv(S)
    det = np.linalg.det(S)

    den = np.sqrt(((2 * np.pi) ** d) * det)

    diff = X.T - m[:, np.newaxis]

    M1 = np.matmul(Q, diff)
    M2 = np.sum(diff * M1, axis=0)
    M3 = - M2 / 2

    density = np.exp(M3) / den

    density = np.where(density < 1e-300, 1e-300, density)

    return density
