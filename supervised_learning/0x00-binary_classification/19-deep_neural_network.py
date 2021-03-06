#!/usr/bin/env python3
"""
defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network Class
    """

    def __init__(self, nx, layers):
        """
        nx is number of input values
        """
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        """
        layers list reping num nodes in each layer
        """
        if type(layers) is not (list) or len(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__nx = nx
        self.__cache = {}
        self.__weights = {}
        for i_lyr in range(self.L):
            mWts = "W" + str(i_lyr + 1)
            mB = "b" + str(i_lyr + 1)
            if type(layers[i_lyr]) is not (int) or layers[i_lyr] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[mB] = np.zeros((layers[i_lyr], 1))
            if i_lyr == 0:
                self.__weights[mWts] = (np.random.randn(layers[i_lyr], nx)
                                        * np.sqrt(2 / nx))
            else:
                self.__weights[mWts] = (np.random.randn(layers[i_lyr],
                                        layers[i_lyr - 1])
                                        * np.sqrt(2 / layers[i_lyr - 1]))

    @property
    def L(self):
        """
        returns length of layers
        """
        return self.__L

    @property
    def nx(self):
        """
        returns number of input values
        """
        return self.__nx

    @property
    def cache(self):
        """
        returns dictionary w/ values of network
        """
        return self.__cache

    @property
    def weights(self):
        """
        return dictionary w/ weights & bias of network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of Neural Network
        """
        """
        X is a numpy.ndarray with shape (nx, m)
        nx is the num of input features to neuron
        m is the num of examples
        """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            idx = layer + 1
            A_prev = self.__cache["A" + str(layer)]
            W_curr = self.__weights["W" + str(idx)]
            b_curr = self.__weights["b" + str(idx)]
            Z = np.matmul(W_curr, A_prev) + b_curr
            self.__cache["A" + str(idx)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates cost of model logistic regression
        """
        """
        Y is a numpy.ndarray with shape (1, m)
        A is a numpy.ndarray with shape (1, m)
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
