#!/usr/bin/env python3
""" Model everithing together """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_layer(prev, n, activation):
    """ creates layers """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor_l = (tf.layers.Dense(units=n, activation=activation,
                kernel_initializer=initialize, name="layer"))
    return tensor_l(prev)


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network
    in tensorflow
    """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = (tf.layers.Dense(units=n, activation=None,
         kernel_initializer=initialize))
    m, s = tf.nn.moments(x(prev), axes=[0], keep_dims=True)
    f = (tf.nn.batch_normalization(x(prev), mean=m, variance=s,
         offset=None, scale=None, variance_epsilon=1e-8))
    return activation(f)


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network """
    net = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        net = create_layer(net, layer_sizes[i], activations[i])
    return net


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    pred = tf.argmax(y_pred, 1)
    val = tf.argmax(y, 1)
    equality = tf.equal(pred, val)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """ calculates the softmax cross-entropy loss of a prediction """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates a learning rate decay operation
    in tensorflow using inverse time decay
    """
    return (tf.train.inverse_time_decay(alpha, global_step,
            decay_step, decay_rate, staircase=True))


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """  builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization
    """
    x = (tf.placeholder(tf.float32,
         shape=(None, Data_train[0].shape[1]), name='x'))
    y = (tf.placeholder(tf.float32,
         shape=(None, Data_train[1].shape[1]), name='y'))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        global_step = tf.Variable(0, trainable=False)
        feed_dict_t = {x: Data_train[0], y: Data_train[1]}
        feed_dict_v = {x: Data_valid[0], y: Data_valid[1]}
        float_iterations = Data_train[0].shape[0]/batch_size
        iterations = int(float_iterations)
        if float_iterations > iterations:
            iterations = int(float_iterations) + 1
            extra = True
        else:
            extra = False
        for epoch in range(epochs + 1):
            cost_t = sess.run(loss, feed_dict_t)
            acc_t = sess.run(accuracy, feed_dict_t)
            cost_v = sess.run(loss, feed_dict_v)
            acc_v = sess.run(accuracy, feed_dict_v)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if epoch < epochs:
                alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
                X_shuffled, Y_shuffled = (shuffle_data(Data_train[0],
                                          Data_train[1]))
                for step in range(iterations):
                    start = step*batch_size
                    if step == iterations - 1 and extra:
                        end = (int(step*batch_size +
                               (float_iterations - iterations + 1) *
                                batch_size))
                    else:
                        end = step*batch_size + batch_size
                    feed_dict_mini = {x: X_shuffled[start: end],
                                      y: Y_shuffled[start: end]}
                    sess.run(train_op, feed_dict_mini)
                    if step != 0 and (step + 1) % 100 == 0:
                        print("\tStep {}:".format(step + 1))
                        cost_mini = sess.run(loss, feed_dict_mini)
                        print("\t\tCost: {}".format(cost_mini))
                        acc_mini = sess.run(accuracy, feed_dict_mini)
                        print("\t\tAccuracy: {}".format(acc_mini))
            save_path = saver.save(sess, save_path)
    return save_path
