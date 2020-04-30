#!/usr/bin/env python3
"""
Create class to represent Poisson distribution
"""


class Poisson:
    """
    Class to represent Poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Data initilization
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return (pow(self.lambtha, k)
                * pow(2.7182818285, -1 * self.lambtha) / factorial(k))


def factorial(m):
    """
    Calculates factorial of a number
    """
    if m == 1 or m == 0:
        return 1
    else:
        return m * factorial(m-1)
