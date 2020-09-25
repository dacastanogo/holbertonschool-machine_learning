#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception block
    @A_prev: output from the previous layer
    @filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        @F1: number of filters in the 1x1 convolution
        @F3R: number of filters in the 1x1 convolution before the
            3x3 convolution
        @F3: number of filters in the 3x3 convolution
        @F5R: number of filters in the 1x1 convolution before the
            5x5 convolution
        @F5 is the number of filters in the 5x5 convolution
        @FPP is the number of filters in the 1x1 convolution after
            the max pooling
    All convolutions inside the inception block uses a rectified
        linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    out_F1 = K.layers.Conv2D(filters=F1,
                             kernel_size=1,
                             padding='same',
                             kernel_initializer=init,
                             activation='relu')(A_prev)

    out_F3R = K.layers.Conv2D(filters=F3R,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init,
                              activation='relu')(A_prev)

    out_F3 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             padding='same',
                             kernel_initializer=init,
                             activation='relu')(out_F3R)

    out_F5R = K.layers.Conv2D(filters=F5R,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init,
                              activation='relu')(A_prev)

    out_F5 = K.layers.Conv2D(filters=F5,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=init,
                             activation='relu')(out_F5R)

    max_pool = K.layers.MaxPool2D(pool_size=3,
                                  strides=1,
                                  padding='same')(A_prev)

    out_FPP = K.layers.Conv2D(filters=FPP,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init,
                              activation='relu')(max_pool)

    output = K.layers.concatenate([out_F1, out_F3, out_F5, out_FPP])
    return output
