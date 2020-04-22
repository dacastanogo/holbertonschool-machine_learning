#!/usr/bin/env python3
"""Adds two matrices
"""


def shape(matrix):
    """
    Getting shape of a matrix
    Return array with dimensions
    """
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def shape_recursion(mat1, mat2):
    """
    Getting shape of a matrix
    Return array with dimensions
    """
    add = []
    for i in range(len(mat1)):
        if type(mat1[i]) == list:
            add.append(shape_recursion(mat1[i], mat2[i]))
        else:
            add.append(mat1[i] + mat2[i])
    return add


def add_matrices(mat1, mat2):
    """
    Adds two matrices
    """
    shape1 = shape(mat1)
    shape2 = shape(mat2)
    tempshape = shape1
    if shape1 != shape2:
        return None
    add = shape_recursion(mat1, mat2)
    return add
