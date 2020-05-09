#!/usr/bin/env python3
"""
class Neuron that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """
    Class defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Updates the private attribute __A
        """
        M = np.matmul(self.__W, X) + self.__b
        self.__A = 1.0/(1.0 + np.exp(-M))
        return self.__A
