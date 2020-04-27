#!/usr/bin/env python3
"""
Calculates the derivative of a polinomy
"""


def poly_derivative(poly):
    """
        Calculates the derivative of a polinomy
        poly: list of integers
        Return: list of integers, or None if poly is invalid
    """
    if not type(poly) is list or len(poly) == 0 or type(poly[0]) is not int:
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    if derivative == []:
        derivative = [0]

    return derivative
