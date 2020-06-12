#!/usr/bin/env python3
""" Convolution with Padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    ph = padding[0]
    pw = padding[1]
    new_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    new_h = new_images.shape[1]
    new_w = new_images.shape[2]
    conv = np.zeros((m, new_h-kh+1, new_w-kw+1))
    img = np.arange(m)
    for j in range(new_h-kh+1):
        for i in range(new_w-kw+1):
            conv[img, j, i] = (np.sum(new_images[img, j:kh+j, i:kw+i] *
                               kernel, axis=(1, 2)))
    return conv
