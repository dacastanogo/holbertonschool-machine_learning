#!/usr/bin/env python3
"""
Function that converts numeric label vector to one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is numpy.ndarray with shape (m,) contains numeric class labels
    m is num of examples
    classes is max num classes found in Y
    Return a one-hot encoding of Y with shape (classes, m),
    or None on failure
    """
    if len(Y) == 0:
        return None
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    else:
        b = np.zeros((classes, Y.shape[0]))
        for clss, m in enumerate(Y):
            b[m][clss] = 1
        return b
