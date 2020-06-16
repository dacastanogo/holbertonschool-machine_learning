#!/usr/bin/env python3
""" Pooling Back Prop """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer of a neural network:

    @dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
        @m: is the number of examples
        @h_new: is the height of the output
        @w_new: is the width of the output
        @c is the number of channels
    @A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
        output of the previous layer
        @h_prev: the height of the previous layer
        @w_prev: the width of the previous layer
    @kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
        @kh is the kernel height
        @kw is the kernel width
    @stride: tuple of (sh, sw) containing the strides for the pooling
        @sh is the stride for the height
        @sw is the stride for the width
    @mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to previous layer (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
    for m_i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c_i in range(c):
                    pool = A_prev[m_i, h*sh:(kh+h*sh), w*sw:(kw+w*sw), c_i]
                    dA_val = dA[m_i, h, w, c_i]
                    if mode == 'max':
                        zero_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(zero_mask, pool == _max, 1)
                        dA_prev[m_i, h*sh:(kh+h*sh),
                                w*sw:(kw+w*sw), c_i] += zero_mask * dA_val
                    if mode == 'avg':
                        avg = dA_val/kh/kw
                        one_mask = np.ones(kernel_shape)
                        dA_prev[m_i, h*sh:(kh+h*sh),
                                w*sw:(kw+w*sw), c_i] += one_mask * avg
    return dA_prev
