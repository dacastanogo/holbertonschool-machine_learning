#!/usr/bin/env python3
"""
Contains  a function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Transpose a  2D matrix
    Returns the transpose of matrix
    """
    transposed_matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transposed_matrix.append(row)
    return(transposed_matrix)
