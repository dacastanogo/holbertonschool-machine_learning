#!/usr/bin/env python3
""" Create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer that includes L2 regularization """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor_l = (tf.layers.Dense(units=n, activation=activation,
                kernel_initializer=initialize, kernel_regularizer=reg))
    return tensor_l(prev)
