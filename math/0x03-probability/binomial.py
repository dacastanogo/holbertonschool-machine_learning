#!/usr/bin/env python3
"""
Create a class Binomial that represents a binomial distribution
"""


class Binomial:
    """
    class Binomial that represents a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Data
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(m - mean) ** 2 for m in data]) / len(data)
            self.p = -1 * (variance / mean - 1)
            n = mean / self.p
            self.n = round(n)
            self.p *= n / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if type(k) is not int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        return (factorial(self.n) / factorial(k) / factorial(self.n - k)
                * self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        acum_prob = 0

        if type(k) is not int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        for m in range(0, k + 1):
            acum_prob += self.pmf(m)
        return acum_prob


def factorial(m):
    """
    Calculates factorial of a number
    """
    if m == 1 or m == 0:
        return 1
    else:
        return m * factorial(m-1)
