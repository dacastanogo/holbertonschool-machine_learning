#!/usr/bin/env python3
"""contains the absorbing function"""

import numpy as np


def route_check(P, n):
    """ check if it is possible to go from each non-absorbing
        state to at least one absorbing state"""

    absorbing_states = np.where(np.diag(P) == 1)

    rows = P[absorbing_states[0]]

    account = np.sum(rows, axis=0)

    for i in range(n):
        row_check = P[i] != 0
        intersection = account * row_check

        # if intersects with a state already accounted for
        if (intersection == 1).any():
            account[i] = 1
    return account.all()


def absorbing(P):
    """
    determines if a markov chain is absorbing
    :param P: P[i, j] is the probability of transitioning
        from state i to state j
        n is the number of states in the markov chain
    :return: True if it is absorbing, or False on failure
    """

    # P conditions
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    if P.shape[0] != P.shape[1]:
        return False

    if np.sum(P, axis=1).all() != 1:
        return False

    n = P.shape[0]

    P_diag = np.diag(P)

    if all(P_diag == 1):
        return True

    if not any(P_diag == 1):
        return False

    return route_check(P, n)
