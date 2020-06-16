#!/usr/bin/env python3
""" Save Only the Best """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
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
        @learning_rate_decay is a boolean that indicates whether
        learning rate decay should be used
        @alpha is the initial learning rate
        @decay_rate is the decay rate
        @save_best boolean indicating whether to save the model
            after each epoch if it is the best
        @filepath is the file path where the model should be saved
        Returns: the History object generated after training the model
    """
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    call_list = []
    if save_best:
        check = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        call_list.append(check)
    if learning_rate_decay and validation_data:
        decay = K.callbacks.LearningRateScheduler(learning_rate, 1)
        call_list.append(decay)
    if early_stopping and validation_data:
        stop_learn = K.callbacks.EarlyStopping(patience=patience)
        call_list.append(stop_learn)
    hist = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                       shuffle=shuffle, verbose=verbose,
                       validation_data=validation_data, callbacks=call_list)
    return hist
