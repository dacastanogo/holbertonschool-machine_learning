#!/usr/bin/env python3
"""
Function that calculates the precision
for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Function that calculates the precision
    for each class in a confusion matrix
    """
    diagonal_tp = np.diagonal(confusion)
    positives = np.sum(confusion, axis=0)
    return diagonal_tp / positives
