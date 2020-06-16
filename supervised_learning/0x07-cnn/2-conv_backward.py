#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:

    @dZ numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output
        of the convolutional layer
        @m is the number of examples
        @h_new is the height of the output
        @w_new is the width of the output
        @c_new is the number of channels in the output
    @A_prev numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        @h_prev is the height of the previous layer
        @w_prev is the width of the previous layer
        @c_prev is the number of channels in the previous layer
    @W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
        @kh is the filter height
        @kw is the filter width
    @b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    @padding is a string that is either same or valid,
        indicating the type of padding used
    @stride is a tuple of (sh, sw) containing the strides for the convolution
        @sh is the stride for the height
        @sw is the stride for the width
    Returns: the partial derivatives with respect to the
        previous layer (dA_prev), the kernels (dW),
        and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    ph = 0
    pw = 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = np.ceil(((sh*h_prev)-sh+kh-h_prev)/2)
        ph = int(ph)
        pw = np.ceil(((sw*w_prev)-sw+kw-w_prev)/2)
        pw = int(pw)

    A_prev = np.pad(A_prev, pad_width=((0, 0),
                    (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dW = np.zeros_like(W)
    dx = np.zeros_like(A_prev)
    for m_i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    tmp_W = W[:, :, :, f]
                    tmp_dz = dZ[m_i, h, w, f]
                    dx[m_i, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += tmp_dz * tmp_W

                    tmp_A_prev = A_prev[m_i, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                    dW[:, :, :, f] += tmp_A_prev * tmp_dz

    dx = dx[:, ph:dx.shape[1]-ph, pw:dx.shape[2]-pw, :]

    return dx, dW, db
