#!/usr/bin/env python3
"""
Create a class Exponential that represents an exponential distribution
"""


class Exponential:
    """
    class Exponential that represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """Initialize data"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        """
        if x < 0:
            return 0
        return self.lambtha * pow(2.7182818285, -1 * self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        """
        if x < 0:
            return 0
        return 1 - pow(2.7182818285, -1 * self.lambtha * x)
