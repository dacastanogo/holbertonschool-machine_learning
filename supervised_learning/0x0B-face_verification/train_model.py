#!/usr/bin/env python3
""" Train Model """
from triplet_loss import TripletLoss
import tensorflow.keras as K
import tensorflow as tf


class TrainModel:
    """ train model """
    def __init__(self, model_path, alpha):
        """  Initialize Train Model
            - model_path is the path to the base face verification
                embedding model
            - loads the model using with
                tf.keras.utils.CustomObjectScope({'tf': tf}):
            - saves this model as the public instance method base_model
            - alpha is the alpha to use for the triplet loss calculation

            Creates a new model:
            inputs: [A, P, N]
                A is a numpy.ndarray containing the anchor images
                P is a numpy.ndarray containing the positive images
                N is a numpy.ndarray containing the negative images
            outputs: the triplet losses of base_model
            compiles the model with Adam optimization and no additional losses
            save this model as the public instance method training_model
            you can use from triplet_loss import TripletLoss
        """

        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.base_model.save(base_model)

        A_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        P_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        N_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        inputs = [A_inputs, P_inputs, N_inputs]
        outputs_embedding = self.base_model(inputs)
        """
        P = self.base_model(P_input)
        N = self.base_model(N_input)
        """
        tl = TripletLoss(alpha)
        output = tl(outputs_embedding)

        training_model = K.models.Model(inputs, output)
        training_model.compile(optimizer='Adam')
        training_model.save('training_model')

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """ trains self.training_model """
        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_split=validation_split)
        return history

    def save(self, save_path):
        """ saves model """
        self.base_model.save(save_path)

    @staticmethod
    def f1_score(y_true, y_pred):
        """calcultes f1 score """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    @staticmethod
    def accuracy(y_true, y_pred):
        """ Calculates Metrics accuracy """
        return K.metrics.accuracy(y_true, y_pred)

    def best_tau(self, images, identities, thresholds):
        """ Best Tau """
