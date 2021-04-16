#!/usr/bin/env python3
"""contains the class BayesianOptimization"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        constructor
        :param f: black-box function to be optimized
        :param X_init: numpy.ndarray of shape (t, 1)
            representing the inputs already sampled with the black-box function
        :param Y_init: numpy.ndarray of shape (t, 1) representing the outputs
            of the black-box function for each input in X_init
        :param bounds: tuple of (min, max) representing the bounds
            of the space in which to look for the optimal point
        :param ac_samples: number of samples that should be analyzed
            during acquisition
        :param l: length parameter for the kernel
        :param sigma_f: standard deviation given to the output of the
            black-box function
        :param xsi: exploration-exploitation factor for acquisition
        :param minimize: bool determining whether optimization
            should be performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        calculates the next best sample location
        :return: X_next, EI
            X_next is a numpy.ndarray of shape (1,)
                representing the next best sample point
            EI is a numpy.ndarray of shape (ac_samples,)
                containing the expected improvement of each potential sample
        """
        X = self.gp.X
        mu_sample, _ = self.gp.predict(X)

        mu, sigma = self.gp.predict(self.X_s)

        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='warn'):
            if self.minimize is True:
                mu_sample_opt = np.amin(self.gp.Y)
                imp = (mu_sample_opt - mu - self.xsi).reshape(-1, 1)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                imp = (mu - mu_sample_opt - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI.reshape(-1)
