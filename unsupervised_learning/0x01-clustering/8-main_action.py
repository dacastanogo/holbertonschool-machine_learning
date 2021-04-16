#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

expectation_maximization = __import__('8-EM_action').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]],
                                      size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]],
                                      size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l_ = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)

    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l_)

    # generate animation with ImageMagick
    os.system('convert -delay 50 frames2/*.png frames2/animation.gif')
