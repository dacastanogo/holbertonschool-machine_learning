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
