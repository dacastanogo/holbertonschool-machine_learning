#!/usr/bin/env python3
""" Gradient Descent with L2 Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    using gradient descent with L2 regularization
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
            dw = np.matmul(dz, cache["A"+str(ly)].T) / m
        dw_reg = dw + (lambtha/m)*tmp_W["W"+str(ly+1)]
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W"+str(ly+1)] = (tmp_W["W"+str(ly+1)] - (alpha * dw_reg))
        weights["b"+str(ly+1)] = tmp_W["b"+str(ly+1)] - alpha * db
        dzp = dz
