#!/usr/bin/env python3
""" Triplet Loss """
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import tensorflow as tf


class TripletLoss(Layer):
    """ Class Triplet Loss """
    def __init__(self, alpha, **kwargs):
        """ Initialize Triplet Loss """
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha


    def triplet_loss(self, inputs):
        """ Calculate Triplet Loss """
        A, P, N = inputs
        distance1 = K.layers.Subtract()([A, P])
        distance2 = K.layers.Subtract()([A, N])

        out1 = K.backend.sum(K.backend.square(distance1), axis=1)
        out2 = K.backend.sum(K.backend.square(distance2), axis=1)

        loss1 = K.layers.Subtract()([out1, out2]) + self.alpha

        loss =  K.backend.maximum(loss1, 0)

        return loss

    def call(self, inputs):
        """ adds the triplet loss to the graph
            - inputs is a list containing the anchor, positive, and negative
                output tensors from the last layer of the model, respectively
        Returns: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
