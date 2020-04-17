#!/usr/bin/env python3
"""
adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    adds two arrays element-wise
    """
    if len(arr1) != len(arr2):
        return

    sum_arrays = []
    for i in range(len(arr1)):
        sum_arrays.append(arr1[i] + arr2[i])

    return(sum_arrays)
