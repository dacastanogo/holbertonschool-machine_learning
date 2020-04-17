#!/usr/bin/env python3
"""
function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    function that performs matrix multiplication
    return multiplicated matrix
    """
    res = []
    if len(mat1[0]) != len(mat2):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat1[0])):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        res.append(row)
    return(res)
