#!/usr/bin/env python3
"""
Calculates the sum of i^2 from i = 1 to n
"""


def summation_i_squared(n):
    """
        Calculates the sum of i^2 from i = 1 to n
        Return: integer, or None if n is negative or 0 or
        not a number
    """
    if type(n) is not int or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
