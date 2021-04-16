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

    # initialization
    i = 0
    P_initial = np.copy(P)
    P_list = [P_initial]

    while True:
        P = np.matmul(P_initial, P)

        for p in P_list:
            if (p == P).all() or i == 1000:
                return None
        if (P > 0).all():
            break

        P_list.append(P)
        i += 1

    # steady state vector calculation
    # sP = s ; s (P - I) = 0
    Q = P_initial - np.identity(n)

    # all probabilities add up to 1
    e = np.ones((1, n))
    b = np.zeros(n)
    b[-1] = 1

    # (P - I).T s = 0
    # replace last row by ones
    Q = np.concatenate((Q.T[:-1], e))

    # solve Qs = b for s
    s = np.linalg.solve(Q, b)

    return s.reshape((1, n))
