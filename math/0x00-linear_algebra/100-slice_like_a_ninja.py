#!/usr/bin/env python3
"""
function slices a matrix along a specific axis
"""


def np_slice(matrix, axes={}):
    """
    function slices a matrix along a specific axis
    Return sliced matrix
    """
    chop = [slice(None)] * (max(axes) + 1)
    for axis, key in axes.items():
        chop[axis] = slice(*key)
    return matrix[tuple(chop)]
