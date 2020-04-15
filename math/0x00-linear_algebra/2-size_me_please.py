#!/usr/bin/env python3
"""
Calculates the shape of a matrix
"""
def matrix_shape(matrix):
    if not matrix:
        return None
    shape = []
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    if (shape.append(len(matrix[0][0]))):
        shape.append(len(matrix[0][0]))
    return(shape)
