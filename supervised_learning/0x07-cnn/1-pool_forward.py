#!/usr/bin/env python3
""" Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network:

    @A_prev numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        @m is the number of examples
        @h_prev is the height of the previous layer
        @w_prev is the width of the previous layer
        @c_prev is the number of channels in the previous layer
    @kernel_shape tuple of (kh, kw) containing the size of the
        kernel for the pooling
        @kh is the kernel height
        @kw is the kernel width
    @stride is a tuple of (sh, sw) containing the strides for the pooling
        @sh is the stride for the height
        @sw is the stride for the width
    @mode is a string containing either max or avg,
        indicating whether to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    out_h = int(((h_prev-kh)/sh) + 1)
    out_w = int(((w_prev-kw)/sw) + 1)
    conv = np.zeros((m, out_h, out_w, c_prev))
    img = np.arange(m)
    for j in range(out_h):
        for i in range(out_w):
            if mode == 'max':
                conv[img, j, i] = (np.max(A_prev[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
            if mode == 'avg':
                conv[img, j, i] = (np.mean(A_prev[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
    return conv
