#!/usr/bin/env python3
"""contains the forward function"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
        calculates the most likely sequence of hidden states for a hidden
        markov model
        :param Observation: numpy.ndarray of shape (T,)
            that contains the index of the observation
            T is the number of observations
        :param Emission: numpy.ndarray of shape (N, M)
            containing the emission probability of a specific
                observation given a hidden state
            Emission[i, j] is the probability of observing j given the
                hidden state i
            N is the number of hidden states
            M is the number of all possible observations
        :param Transition: 2D numpy.ndarray of shape (N, N)
            containing the transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        :param Initial: numpy.ndarray of shape (N, 1) containing the
            probability of starting
            in a particular hidden state
        :return: path, P, or None, None on failure
            path is the a list of length T containing the most likely
            sequence of hidden states
            P is the probability of obtaining the path sequence
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    # initialization
    Obs_t = Observation[0]
    backpointer[:, 0] = 0
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_t])
    viterbi[:, 0] = prob

    # recursion
    for t in range(1, T):
        a = viterbi[:, t - 1]
        b = Transition.T
        ab = a * b
        ab_max = np.amax(ab, axis=1)
        c = Emission[:, Observation[t]]
        prob = ab_max * c

        viterbi[:, t] = prob
        backpointer[:, t - 1] = np.argmax(ab, axis=1)

    # path initialization
    path = []
    current = np.argmax(viterbi[:, T - 1])
    path = [current] + path

    # path backwards traversing
    for t in range(T - 2, -1, -1):
        current = int(backpointer[current, t])
        path = [current] + path

    # max path probabilities among all possible states
    # end of path

    P = np.amax(viterbi[:, T - 1], axis=0)

    return path, P
