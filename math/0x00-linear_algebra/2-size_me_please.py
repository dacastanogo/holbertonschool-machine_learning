#!/usr/bin/env python3
"""
Calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Getting shape of a matrix
    Return array with dimensions
    """
    res = []
    shape_recursion(res, matrix)
    return(res)


def shape_recursion(res, matrix):
    """
    Getting shape of a matrix
    Return array with dimensions
    """
    if type(matrix) is int:
        return
    for arr in matrix:
        res.append((len(matrix)))
        return(shape_recursion(res, arr))
