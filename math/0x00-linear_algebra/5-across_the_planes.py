#!/usr/bin/env python3
"""
adds two same shape matrix element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    adds two same shape matrix element-wise
    return the added matrix
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return
    added_matrix = []
    for x in range(len(mat1)):
        row = []
        for i in range(len(mat1[0])):
            row.append(mat1[x][i] + mat2[x][i])
        added_matrix.append(row)
    return(added_matrix)


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
