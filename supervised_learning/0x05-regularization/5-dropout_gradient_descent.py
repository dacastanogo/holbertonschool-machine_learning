#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a neural network with Dropout
    regularization using gradient descent
    """
    tmp_W = weights.copy()
    m = Y.shape[1]
    for ly in reversed(range(L)):
        if ly == L-1:
            dz = cache["A"+str(ly+1)] - Y
            dw = (np.matmul(cache["A"+str(ly)], dz.T) / m).T
        else:
            d1 = np.matmul(tmp_W["W"+str(ly+2)].T, dzp)
            d2 = 1-cache["A"+str(ly+1)]**2
            dz = d1 * d2
            dz *= cache["D"+str(ly+1)]
            dz /= keep_prob
            dw = np.matmul(dz, cache["A"+str(ly)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W"+str(ly+1)] = tmp_W["W"+str(ly+1)] - alpha * dw
        weights["b"+str(ly+1)] = tmp_W["b"+str(ly+1)] - alpha * db
        dzp = dz
