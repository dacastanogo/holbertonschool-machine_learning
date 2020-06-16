#!/usr/bin/env python3
""" Early Stopping """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent and validate it
        trains the model using early stopping

        @network is the model to train
        @data numpy.ndarray (m, nx) containing the input data
        @labels one-hot numpy.ndarray (m, classes) containing
                the labels of data
        @batch_size size of the batch used for mini-batch GD
        @epochs number of passes through data for mini-batch GD
        @verbose boolean that determines if output should be printed
        @shuffle boolean that determines whether to shuffle the batches
                every epoch.
        @validation_data data to validate the model with, if not None
        early_stopping boolean indicates whether early stopping should be used
                should only be performed if validation_data exists
                should be based on validation loss
        @patience is the patience used for early stopping
        Returns: the History object generated after training the model
    """
    stop_learn = None
    if early_stopping and validation_data:
        stop_learn = [K.callbacks.EarlyStopping(patience=patience)]
    hist = network.fit(x=data, y=labels, epochs=epochs, batch_size=batch_size,
                       shuffle=shuffle, verbose=verbose,
                       validation_data=validation_data, callbacks=stop_learn)
    return hist
