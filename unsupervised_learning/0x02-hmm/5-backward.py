#!/usr/bin/env python3
"""contains the backward function"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model
    :param Observation: numpy.ndarray of shape (T,)
        that contains the index of the observation
        T is the number of observations
    :param Emission: numpy.ndarray of shape (N, M)
        containing the emission probability of a specific observation
        given a hidden state
        Emission[i, j] is the probability of observing j given the
        hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    :param Transition: 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j
    :param Initial: numpy.ndarray of shape (N, 1) containing the probability
        of starting
        in a particular hidden state
    :return: P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the
        backward path probabilities
        B[i, j] is the probability of generating the future observations
        from hidden state i at time j
    """
    # type and len(dim) conditions
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    # dim conditions
    T = Observation.shape[0]

    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # stochastic
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    Beta = np.zeros((N, T))
    # initialization
    Beta[:, T - 1] = np.ones(N)

    # recursion
    for t in range(T - 2, -1, -1):
        a = Transition
        b = Emission[:, Observation[t + 1]]
        c = Beta[:, t + 1]

        abc = a * b * c
        prob = np.sum(abc, axis=1)
        Beta[:, t] = prob

    # sum of path probabilities over all possible states
    # end of path
    P_first = Initial[:, 0] * Emission[:, Observation[0]] * Beta[:, 0]
    P = np.sum(P_first)

    return P, Beta
