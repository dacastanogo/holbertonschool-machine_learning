#!/usr/bin/env python3
""" Convolutional Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer
        of a neural network

        @A_prev numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            @m is the number of examples
            @h_prev is the height of the previous layer
            @w_prev is the width of the previous layer
            @c_prev is the number of channels in the previous layer
        @W numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels
            @kh is the filter height
            @kw is the filter width
            @c_prev is the number of channels in the previous layer
            @c_new is the number of channels in the output
        @b numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        @activation is an activation function applied to the convolution
        @padding string that is either same or valid,
            indicating the type of padding used
        @stride is a tuple of (sh, sw) containing the strides
            @sh is the stride for the height
            @sw is the stride for the width
        Returns: the output of the convolutional layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]
    sh = stride[0]
    sw = stride[1]
    ph = 0
    pw = 0
    conv_h = int(((h_prev+2*ph-kh)/sh) + 1)
    conv_w = int(((w_prev+2*pw-kw)/sw) + 1)
    if padding == 'same':
        if kh % 2 == 0:
            ph = int(((h_prev)*sh+kh-h_prev)/2)
            conv_h = int(((h_prev+2*ph-kh)/sh))
        else:
            ph = int(((h_prev-1)*sh+kh-h_prev)/2)
            conv_h = int(((h_prev+2*ph-kh)/sh)+1)
        if kw % 2 == 0:
            pw = int(((w_prev)*sw+kw-w_prev)/2)
            conv_w = int(((w_prev+2*pw-kw)/sw))
        else:
            pw = int(((w_prev-1)*sw+kw-w_prev)/2)
            conv_w = int(((w_prev+2*pw-kw)/sw)+1)
        A_prev = np.pad(A_prev, pad_width=((0, 0),
                        (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    conv = np.zeros((m, conv_h, conv_w, c_new))
    img = np.arange(m)
    for j in range(conv_h):
        for i in range(conv_w):
            for k in range(c_new):
                tmp = A_prev[img, j*sh:(kh+(j*sh)), i*sw:(kw+(i*sw))]
                conv[img, j, i, k] = (np.sum(tmp *
                                      W[:, :, :, k],
                                      axis=(1, 2, 3)))
                conv[img, j, i, k] = (activation(conv[img, j, i, k] +
                                      b[0, 0, 0, k]))
    return conv
