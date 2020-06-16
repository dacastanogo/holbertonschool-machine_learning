#!/usr/bin/env python3
""" Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire model
        @network model to save
        @filename path of the file
        Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """ loads an entire model
    @filename path of the file
    Returns: the loaded model
    """
    return K.models.load_model(filename)
