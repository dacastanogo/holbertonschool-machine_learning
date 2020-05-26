#!/usr/bin/env python3
"""
function that calculates the specificity
for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    function that calculates the specificity
    for each class in a confusion matrix
    """
    diagonal = np.diagonal(confusion)
    fullset = np.sum(confusion)
    fullset_array = np.full_like(confusion[0], fullset)
    cross1 = [np.sum(i) for i in confusion]
    cross2 = [np.sum(i) for i in confusion.T]
    tn = fullset_array - cross1 - cross2 + diagonal
    fp = np.sum(confusion, axis=0) - np.diagonal(confusion)
    return tn / (fp+tn)
