#!/usr/bin/env python3
""" Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images:

        @images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            @m is the number of images
            @h is the height in pixels of the images
            @w is the width in pixels of the images
        @kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
        @kh is the height of the kernel
        @kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = int((kh - 1)/2)
    pw = int((kw - 1)/2)
    if kh % 2 == 0:
        ph = int(kh/2)
    if kw % 2 == 0:
        pw = int(kw/2)
    new_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    conv = np.zeros((m, h, w))
    img = np.arange(m)
    for j in range(h):
        for i in range(w):
            conv[img, j, i] = (np.sum(new_images[img, j:kh+j, i:kw+i] *
                               kernel, axis=(1, 2)))
    return conv
