#!/usr/bin/env python3
"""contain the sparse function"""

import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder
    :param input_dims: integer containing the dimensions of the model input
    :param hidden_layers: list containing the number of nodes
        for each hidden layer in the encoder, respectively
    :param latent_dims:  integer containing the dimensions
        of the latent space representation
    :param lambtha: regularization parameter used for
        L1 regularization on the encoded output
    :return: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model
    """
    # ************************************************************
    # ENCODER
    # input placeholder
    inputs = keras.Input(shape=(input_dims,))

    # regularization scheme
    reg = keras.regularizers.l1(lambtha)

    # first densely-connected layer
    my_layer = keras.layers.Dense(units=hidden_layers[0],
                                  activation='relu',
                                  activity_regularizer=reg,
                                  input_shape=(input_dims,))(inputs)

    # subsequent densely-connected layers:
    for i in range(1, len(hidden_layers)):
        my_layer = keras.layers.Dense(units=hidden_layers[i],
                                      activity_regularizer=reg,
                                      activation='relu'
                                      )(my_layer)

    # last layer
    my_layer = keras.layers.Dense(units=latent_dims,
                                  activity_regularizer=reg,
                                  activation='relu'
                                  )(my_layer)

    encoder = keras.Model(inputs=inputs, outputs=my_layer)

    # ************************************************************
    # DECODER
    # input placeholder
    inputs_dec = keras.Input(shape=(latent_dims,))

    # first densely-connected layer
    my_layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                      activation='relu',
                                      input_shape=(latent_dims,))(inputs_dec)

    # subsequent densely-connected layers:
    for i in range(len(hidden_layers) - 2, -1, -1):
        my_layer_dec = keras.layers.Dense(units=hidden_layers[i],
                                          activation='relu'
                                          )(my_layer_dec)

    #  last layer in the decoder
    my_layer_dec = keras.layers.Dense(units=input_dims,
                                      activation='sigmoid'
                                      )(my_layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=my_layer_dec)

    # ************************************************************
    # AUTOENCODER
    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)

    # compilation
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
