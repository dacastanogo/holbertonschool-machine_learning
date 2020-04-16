#!/usr/bin/env python3
"""
Calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Getting shape of a matrix
    Return array with dimensions
    """
    if not matrix:
        return None
    shape = []
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    print ( "len after 2 dimensions is: {}".format(len(matrix[0])))
    if (len(matrix[0])) > 2:
        shape.append(len(matrix[0][0]))
    return(shape)
