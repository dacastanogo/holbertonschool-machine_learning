#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

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
    kmin = 1
    kmax = 10

    best_k, best_result, l_, b = BIC(X, kmin=kmin, kmax=kmax)
    print(best_k)
    print(best_result)
    print(l_)
    print(b)
    x = np.arange(kmin, kmax + 1)

    fig = plt.figure()
    plt.plot(x, l_, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    fig.savefig('images/9a-BIC.jpg')

    fig = plt.figure()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
    fig.savefig('images/9b-BIC.jpg')
