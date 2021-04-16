#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

kmeans = __import__('1-kmeans_action').kmeans


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]],
                                      size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 6)
    print(C)

    # generate animation with ImageMagick
    os.system('convert -delay 30 frames/*.png frames/animation.gif')
