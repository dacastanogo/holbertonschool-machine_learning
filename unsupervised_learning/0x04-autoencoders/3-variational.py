#!/usr/bin/env python3
"""contain the autoencoder function"""

import tensorflow.keras as keras


def sampling(args):
    """
    re-parametrization to enable back propagation
    :param args:
        mu: mean from previous layer
        sigma: std from previous layer
    :return: z
        distribution sample
    """
    # unpacking
    mu, sigma = args

    # dimension for normal distribution same as z_mean
    m = keras.backend.shape(mu)[0]
    dims = keras.backend.int_shape(mu)[1]

    # sampling from a normal distribution with mean=0 and standard deviation=1
    # epsilon ~ N(0,1)
    epsilon = keras.backend.random_normal(shape=(m, dims))

    # sampled vector
    z = mu + keras.backend.exp(0.5 * sigma) * epsilon

    return z


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder
    :param input_dims: integer containing the dimensions of the model input
    :param hidden_layers: list containing the number of nodes for each
        hidden layer in the encoder, respectively
    :param latent_dims:  integer containing the dimensions of the latent
        space representation
    :return: Returns: encoder, decoder, auto
        encoder is the encoder model, which should output
            the latent representation, the mean, and the log variance
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # *************************************************************************
    # ENCODER
    # input placeholder
    inputs = keras.Input(shape=(input_dims,))

    # first densely-connected layer
    my_layer = keras.layers.Dense(units=hidden_layers[0],
                                  activation='relu',
                                  input_shape=(input_dims,))(inputs)

    # subsequent densely-connected layers:
    for i in range(1, len(hidden_layers)):
        my_layer = keras.layers.Dense(units=hidden_layers[i],
                                      activation='relu'
                                      )(my_layer)

    # split into mean and sigma layers
    mu = keras.layers.Dense(units=latent_dims)(my_layer)
    sigma = keras.layers.Dense(units=latent_dims)(my_layer)

    # sampling layer
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, sigma])

    encoder = keras.Model(inputs=inputs, outputs=[z, mu, sigma])
    encoder.summary()

    # *************************************************************************
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
    decoder.summary()

    # *************************************************************************
    # AUTOENCODER
    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)
    auto.summary()

    def custom_loss(loss_input, loss_output):
        """ custom loss function """
        # Reconstruction loss
        reconstruction_i = keras.backend.binary_crossentropy(loss_input,
                                                             loss_output)
        reconstruction_sum = keras.backend.sum(reconstruction_i, axis=1)

        # Kullback–Leibler divergence
        kl_i = keras.backend.square(sigma) \
            + keras.backend.square(mu) \
            - keras.backend.log(1e-8 + keras.backend.square(sigma)) \
            - 1

        kl_sum = 0.5 * keras.backend.sum(kl_i, axis=1)

        return reconstruction_sum + kl_sum

    # compilation
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss=custom_loss)

    return encoder, decoder, auto
