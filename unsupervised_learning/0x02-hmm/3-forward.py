#!/usr/bin/env python3
"""contains the forward function"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model
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
    :return: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T)
        containing the forward path probabilities
        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
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

    F = np.zeros((N, T))

    # initialization
    Obs_i = Observation[0]
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_i])
    F[:, 0] = prob

    # recursion
    for i in range(1, T):
        Obs_i = Observation[i]
        state = np.matmul(F[:, i - 1], Transition)
        """
        # equivalent to matmul
        a = F[:, i - 1]
        b = Transition.T
        ab = a * b
        ab_sum = np.sum(ab, axis=1)
        """
        prob = np.multiply(state, Emission[:, Obs_i])
        F[:, i] = prob

    # sum of path probabilities over all possible states
    # end of path
    P = np.sum(F[:, T - 1])

    return P, F
