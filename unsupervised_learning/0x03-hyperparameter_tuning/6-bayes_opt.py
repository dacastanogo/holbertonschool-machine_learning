#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization


def f(X, noise=0.2):
    return -np.sin(3 * X) - X ** 2 + 0.7 * X + noise * np.random.randn(*X.shape)


if __name__ == '__main__':
    np.random.seed(0)
    bounds = np.array([[-1.0, 2.0]])
    noise = 0.2

    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

    # Noise-free objective function values at X
    Y = f(X, 0)

    # Plot optimization objective with noise level
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()
    plt.show()

    # ***************************

    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]

    optimizer = BayesianOptimization(f=f,
                                     domain=bds,
                                     model_type='GP',
                                     kernel=kernel,
                                     acquisition_type='EI',
                                     acquisition_jitter=0.01,
                                     X=X_init,
                                     Y=-Y_init,
                                     noise_var=noise ** 2,
                                     exact_feval=False,
                                     normalize_Y=False,
                                     maximize=True)

    optimizer.run_optimization(max_iter=10)
    optimizer.plot_acquisition()
    plt.show()

    optimizer.plot_convergence()
    plt.show()
