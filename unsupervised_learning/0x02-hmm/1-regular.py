#!/usr/bin/env python3
"""contains the regular function"""

import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    :param P: square 2D numpy.ndarray of shape (n, n)
    representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    :return: numpy.ndarray of shape (1, n)
        containing the steady state probabilities,
        or None on failure
    """
    # P conditions
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    n = P.shape[0]

    # s P = s ; P.T s.T = s.T
    # eigenvalues, eigenvectors calculation
    w, v = np.linalg.eig(P.T)

    # index for eigenvalue equal to one
    index = np.where(np.isclose(w, 1))

    # check if any found
    if len(index[0]):
        index = index[0][0]
    else:
        return None

    # get corresponding eigenvector
    s = v[:, index]

    # check for any zero element
    if any(np.isclose(s, 0)):
        return None

    # normalize s
    s = s / np.sum(s)

    # reshape s
    s = s[np.newaxis, :]

    return s
