#!/usr/bin/env python3
""" Test """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ tests a neural network:

    @network network model to test
    @data input data to test the model with
    @labels correct one-hot labels of data
    @verbose is a boolean that determines if output should be printed
        during the testing
    Returns: loss and accuracy of the model with the testing data
    """
    result = network.evaluate(data, labels, verbose=verbose)
    return result
