#!/usr/bin/env python3
""" Create a Layer with Dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates a layer of a neural network using dropout """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor_l = (tf.layers.Dense(units=n, activation=activation,
                kernel_initializer=initialize))
    drop = tf.layers.Dropout(keep_prob)
    return drop(tensor_l(prev))
