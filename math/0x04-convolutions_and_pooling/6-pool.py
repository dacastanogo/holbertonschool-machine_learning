#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs a convolution on images with channels:

        @images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            @m is the number of images
            @h is the height in pixels of the images
            @w is the width in pixels of the images
            @c is the number of channels in the image
        @kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
        @kh is the height of the kernel
        @kw is the width of the kernel
        @padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                @ph is the padding for the height of the image
                @pw is the padding for the width of the image
        @stride is a tuple of (sh, sw)
            @sh is the stride for the height of the image
            @sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    out_h = int(((h-kh)/sh) + 1)
    out_w = int(((w-kw)/sw) + 1)
    conv = np.zeros((m, out_h, out_w, c))
    img = np.arange(m)
    for j in range(out_h):
        for i in range(out_w):
            if mode == 'max':
                conv[img, j, i] = (np.max(images[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
            if mode == 'avg':
                conv[img, j, i] = (np.mean(images[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
    return conv
