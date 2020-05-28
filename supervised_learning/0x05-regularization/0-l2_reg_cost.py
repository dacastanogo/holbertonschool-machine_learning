#!/usr/bin/env python3
"""
Function that calculates the cost of a
neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a
    neural network with L2 regularization
    """
    Frobenius = 0
    for k, v in weights.items():
        if k[0] == "W":
            Frobenius += np.linalg.norm(v)
    return cost + (lambtha/(2*m)) * Frobenius
