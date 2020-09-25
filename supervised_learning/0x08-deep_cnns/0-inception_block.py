#!/usr/bin/env python3
"""
inception block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in Going Deeper with
        Convolutions (2014): https://arxiv.org/pdf/1409.4842.pdf
        Args:
            - A_prev is the output from the previous layer
            - filters: is a tuple or list containing F1, F3R, F3,F5R, F5,
                FPP, respectively:
                - F1: is the number of filters in the 1x1 convolution
                - F3R: is the number of filters in the 1x1 convolution before
                    the 3x3 convolution
                - F3: is the number of filters in the 3x3 convolution
                - F5R: is the number of filters in the 1x1 convolution before
                    the 5x5 convolution
                - F5: is the number of filters in the 5x5 convolution
                - FPP: is the number of filters in the 1x1 convolution after
                    the max pooling
            - All convolutions inside the inception block should use a
                rectified linear activation (ReLU)
        Returns:
            the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.he_normal(seed=None)

    # 1x1 convolution
    F1_conv = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding='same',
                              kernel_initializer=initializer,
                              activation='relu')(A_prev)

    # 1x1 convolution before the 3x3 convolution
    F3R_conv = K.layers.Conv2D(filters=F3R,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')(A_prev)

    # 3x3 convolution
    F3_conv = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=initializer,
                              activation='relu')(F3R_conv)

    # 1x1 convolution before the 5x5 convolution
    F5R_conv = K.layers.Conv2D(filters=F5R,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')(A_prev)

    # 5x5 convolution
    F5_conv = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5),
                              padding='same',
                              kernel_initializer=initializer,
                              activation='relu')(F5R_conv)

    # max pooling
    max_pooling = K.layers.MaxPooling2D(pool_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same')(A_prev)

    # 1x1 convolution after the max pooling
    FPP_conv = K.layers.Conv2D(filters=FPP,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')(max_pooling)

    # concatenated output of the inception block
    inception_block = K.layers.concatenate([F1_conv,
                                            F3_conv,
                                            F5_conv,
                                            FPP_conv],
                                           axis=-1)

    return inception_block
