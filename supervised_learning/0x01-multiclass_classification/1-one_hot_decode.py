#!/usr/bin/env python3
"""
Function that converts one-hot matrix to vector of labels
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot is encoded numpy.ndarray w/ shape
    (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    Return numpy.ndarray with shape (m, ) containing
    the numeric labels for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2 or len(one_hot) == 0:
        return None
    if type(one_hot) is not np.ndarray:
        return None
    else:
        return np.argmax(one_hot, axis=0)
