#!/usr/bin/env python3
"""contains the class GaussianProcess"""

import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        constructor
        :param X_init: numpy.ndarray of shape (t, 1)
            representing the inputs already sampled with the black-box function
        :param Y_init: numpy.ndarray of shape (t, 1)
            representing the outputs of the black-box function for each input
            in X_init
            t is number of initial samples
        :param l: length parameter for the kernel
        :param sigma_f: standard deviation given to the output of
            the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices:
        :param X1: numpy.ndarray of shape (m, 1)
        :param X2: numpy.ndarray of shape (n, 1)
        :return: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) \
            + np.sum(X2 ** 2, 1) \
            - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points in a
        Gaussian process
        :param X_s: numpy.ndarray of shape (s, 1)
            containing all of the points whose mean and
            standard deviation should be calculated
            s is the number of sample points
        :return: mu, sigma
            mu is a numpy.ndarray of shape (s,)
                containing the mean for each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,)
                containing the standard deviation for each point in X_s
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Equation (4)
        mu = K_s.T.dot(K_inv).dot(self.Y)[:, 0]

        # Equation (5)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process
        :param X_new: numpy.ndarray of shape (1,) that
            represents the new sample point
        :param Y_new: numpy.ndarray of shape (1,) that
            represents the new sample function value
        Updates the public instance attributes X, Y, and K
        """
        self.X = np.append(self.X, X_new)
        self.X = self.X[:, np.newaxis]

        self.Y = np.append(self.Y, Y_new)
        self.Y = self.Y[:, np.newaxis]

        self.K = self.kernel(self.X, self.X)
