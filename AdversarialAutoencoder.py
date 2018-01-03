"""
    Implementation of an Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""

import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin
import collections
import DataLoading


# def get_predictions(logits):
#     condition = tf.less(logits, 0.0)
#     return tf.where(condition, tf.zeros_like(logits), tf.ones_like(logits))


class AdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 input_dim: int,
                 z_dim: int,
                 batch_size: int,
                 n_epochs: int,
                 n_neurons_of_hidden_layer_x_autoencoder: [],
                 n_neurons_of_hidden_layer_x_discriminator: [],
                 bias_init_value_of_hidden_layer_x_autoencoder: [],
                 bias_init_value_of_hidden_layer_x_discriminator: [],
                 learning_rate_autoencoder=0.001, learning_rate_discriminator=0.001, learning_rate_generator=0.001,
                 beta1_autoencoder=0.9, beta1_discriminator=0.9, beta1_generator=0.9,
                 beta2_autoencoder=0.9, beta2_discriminator=0.9, beta2_generator=0.9,
                 loss_function_discriminator="sigmoid_cross_entropy",
                 loss_function_generator="sigmoid_cross_entropy",
                 results_path='./Results/AdvAutoencoder'):
        """
        constructor
        :param input_dim: input dimension of the data
        :param z_dim: dimension of the latent reprensentation
        :param batch_size: number of training examples in one forward/backward pass
        :param n_epochs: number of epochs for training
        :param n_neurons_of_hidden_layer_x_autoencoder: array holding the number of neurons for the layers of the
        autoencoder; position x in array corresponds to layer x+1
        :param n_neurons_of_hidden_layer_x_discriminator: array holding the number of neurons for the layers of the
        discriminator; position x in array corresponds to layer x+1
        :param bias_init_value_of_hidden_layer_x_autoencoder: array holding the initial bias values for the layers of
        the autoencoder; position x in array corresponds to layer x+1
        :param bias_init_value_of_hidden_layer_x_discriminator: array holding the initial bias values for the layers of
        the discriminator; position x in array corresponds to layer x+1
        :param learning_rate_autoencoder: learning rate of the autoencoder optimizer
        :param learning_rate_discriminator: learning rate of the discriminator optimizer
        :param learning_rate_generator: learning rate of the generator optimizer
        :param beta1_autoencoder: exponential decay rate for the 1st moment estimates for the adam optimizer for the
        autoencoder.
        :param beta1_discriminator: exponential decay rate for the 1st moment estimates for the adam optimizer for the
        discriminator.
        :param beta1_generator: exponential decay rate for the 1st moment estimates for the adam optimizer for the
        generator.
        :param beta2_autoencoder: exponential decay rate for the 2nd moment estimates for the adam optimizer for the
        autoencoder.
        :param beta2_discriminator: exponential decay rate for the 2nd moment estimates for the adam optimizer for the
        discriminator.
        :param beta2_generator: exponential decay rate for the 2nd moment estimates for the adam optimizer for the
        generator.
        :param results_path: path where to store the log, the saved models and the tensorboard event file
        """

        self.input_dim = input_dim
        self.z_dim = z_dim

        """
        params for network topology
        """

        # number of neurons of the hidden layers
        self.n_neurons_of_hidden_layer_x_autoencoder = n_neurons_of_hidden_layer_x_autoencoder
        self.n_neurons_of_hidden_layer_x_discriminator = n_neurons_of_hidden_layer_x_discriminator

        # initial bias values of the hidden layers
        self.bias_init_value_of_hidden_layer_x_autoencoder = bias_init_value_of_hidden_layer_x_autoencoder
        self.bias_init_value_of_hidden_layer_x_discriminator = bias_init_value_of_hidden_layer_x_discriminator

        """
        params for learning
        """

        # number of epochs for training
        self.n_epochs = n_epochs

        # number of training examples in one forward/backward pass
        self.batch_size = batch_size

        # learning rate for the different parts of the network
        self.learning_rate_autoencoder = learning_rate_autoencoder
        self.learning_rate_discriminator = learning_rate_discriminator
        self.learning_rate_generator = learning_rate_generator

        # exponential decay rate for the 1st moment estimates for the adam optimizer.
        self.beta1_autoencoder = beta1_autoencoder
        self.beta1_discriminator = beta1_discriminator
        self.beta1_generator = beta1_generator

        # exponential decay rate for the 2nd moment estimates for the adam optimizer.
        self.beta2_autoencoder = beta2_autoencoder
        self.beta2_discriminator = beta2_discriminator
        self.beta2_generator = beta2_generator

        # loss function
        self.loss_function_discriminator = loss_function_discriminator
        self.loss_function_generator = loss_function_generator

        # path for the results
        self.results_path = results_path

        # holds the input data
        self.X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
        # holds the desired output of the autoencoder
        self.X_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
        # holds the real distribution p(z) used as positive sample for the discriminator
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
        # holds the input samples for the decoder (only for generating the images; NOT used for training)
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')

        """
        Init the network; generator doesn't need to be initiated, since the generator is the encoder of the autoencoder
        """

        # init autoencoder
        with tf.variable_scope(tf.get_variable_scope()):
            # encoder part of the autoencoder and also the generator
            encoder_output = self.encoder(self.X, bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # decoder part of the autoencoder
            decoder_output = self.decoder(encoder_output,
                                          bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        # init discriminator
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive samples p(z) (from a real data distribution)
            discriminator_pos_samples = \
                self.discriminator(self.real_distribution,
                                   bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)
            # discriminator for the negative samples q(z) (generated by the generator)
            discriminator_neg_samples = \
                self.discriminator(encoder_output, reuse=True,
                                   bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)

        # TODO: only for testing
        self.discriminator_neg_samples_ = discriminator_neg_samples
        self.discriminator_pos_samples_ = discriminator_pos_samples

        # output of the decoder
        with tf.variable_scope(tf.get_variable_scope()):
            # decoder output: only used for generating the images for tensorboard
            self.decoder_output = self.decoder(self.decoder_input, reuse=True,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        """
        Init the loss functions
        """

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - decoder_output))

        # Discriminator Loss
        discriminator_loss_pos_samples = tf.reduce_mean(
            self.get_loss_function(loss_function=self.loss_function_discriminator,
                                   labels=tf.ones_like(discriminator_pos_samples),
                                   logits=discriminator_pos_samples))
        discriminator_loss_neg_samples = tf.reduce_mean(
            self.get_loss_function(loss_function=self.loss_function_discriminator,
                                   labels=tf.zeros_like(discriminator_neg_samples),
                                   logits=discriminator_neg_samples))
        self.discriminator_loss = discriminator_loss_neg_samples + discriminator_loss_pos_samples

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            self.get_loss_function(loss_function=self.loss_function_generator,
                                   labels=tf.ones_like(discriminator_neg_samples), logits=discriminator_neg_samples))

        """
        Init the optimizers
        """

        all_variables = tf.trainable_variables()
        discriminator_vars = [var for var in all_variables if 'discriminator_' in var.name]
        encoder_vars = [var for var in all_variables if 'encoder_' in var.name]

        # Optimizers
        self.autoencoder_optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate_autoencoder,
                                   beta1=self.beta1_autoencoder,
                                   beta2=self.beta2_autoencoder).minimize(self.autoencoder_loss)
        self.discriminator_optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate_discriminator,
                                   beta1=self.beta1_discriminator,
                                   beta2=self.beta2_discriminator).minimize(self.discriminator_loss,
                                                                            var_list=discriminator_vars)
        self.generator_optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate_generator,
                                   beta1=self.beta1_generator,
                                   beta2=self.beta2_generator).minimize(self.generator_loss,
                                                                        var_list=encoder_vars)

        """
        Create the tensorboard summary
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=decoder_output, encoder_output=encoder_output,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_loss=self.discriminator_loss,
                                            generator_loss=self.generator_loss,
                                            real_distribution=self.real_distribution)

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    @staticmethod
    def get_loss_function(loss_function, labels, logits):
        """
        returns the respective tensorflow loss function
        https://www.tensorflow.org/api_guides/python/contrib.losses#Loss_operations_for_use_in_neural_networks_
        :param loss_function: tensorflow loss function to return
        :param labels: labels
        :param logits: [batch_size, num_classes] logits outputs of the network
        :return:
        """

        return {
            # "absolute_difference": tf.losses.absolute_difference(
            #     labels=labels, predictions=logits),
            "hinge_loss": tf.losses.hinge_loss(
                labels=labels, logits=logits),
            # "log_loss": tf.losses.log_loss(
            #     labels=labels, predictions=logits),#get_predictions(logits)),
            # "mean_pairwise_squared_error": tf.losses.mean_pairwise_squared_error(
            #     labels=labels, predictions=logits),
            "mean_squared_error": tf.losses.mean_squared_error(
                labels=labels, predictions=logits),
            "sigmoid_cross_entropy": tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits),
            "softmax_cross_entropy": tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits),
        }[loss_function]

    @staticmethod
    def create_dense_layer(X, n_input_neurons, n_output_neurons, variable_scope_name, bias_init_value=0.0):
        """
        Used to create a dense layer.
        :param X: input tensor to the dense layer
        :param n_input_neurons: no. of input neurons
        :param n_output_neurons: no. of output neurons
        :param variable_scope_name: name of the entire dense layer
        :param bias_init_value: the initialisation value for the bias
        :return: tensor with shape [batch_size, n2]
        """
        with tf.variable_scope(variable_scope_name, reuse=None):
            weights = tf.get_variable("weights", shape=[n_input_neurons, n_output_neurons],
                                      initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[n_output_neurons],
                                   initializer=tf.constant_initializer(bias_init_value))
            out = tf.add(tf.matmul(X, weights), bias, name='matmul')
            return out

    def encoder(self, X, bias_init_values, reuse=False):
        """
        Encoder of the autoencoder.
        :param X: input to the autoencoder
        :param bias_init_values: the initial value for the bias
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """

        # number of hidden layers
        n_hidden_layers = len(self.n_neurons_of_hidden_layer_x_autoencoder)

        assert n_hidden_layers == len(bias_init_values) - 1

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Encoder'
        with tf.name_scope('Encoder'):
            # there is no hidden layer
            # todo: reasonable default values
            if n_hidden_layers == 0:
                latent_variable = self.create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                                          bias_init_value=bias_init_values[0])
                return latent_variable
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    self.create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                latent_variable = self.create_dense_layer(dense_layer_1,
                                                          self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                          self.z_dim, 'encoder_output',
                                                          bias_init_value=bias_init_values[1])
                return latent_variable
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    self.create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n_hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        self.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                'encoder_dense_layer_' + str(i + 1),
                                                bias_init_value=bias_init_values[i]))
                latent_variable = self.create_dense_layer(dense_layer_i,
                                                          self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                          self.z_dim, 'encoder_output',
                                                          bias_init_value=bias_init_values[-1])
                return latent_variable

    def decoder(self, X, bias_init_values, reuse=False):
        """
        Decoder of the autoencoder.
        :param X: input to the decoder
        :param bias_init_values: the initial values for the bias
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        # number of hidden layers
        n_hidden_layers = len(self.n_neurons_of_hidden_layer_x_autoencoder)

        assert n_hidden_layers == len(bias_init_values) - 1

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Decoder'
        with tf.name_scope('Decoder'):
            # there is no hidden layer
            if n_hidden_layers == 0:
                decoder_output = tf.nn.sigmoid(
                    self.create_dense_layer(X, self.z_dim, self.input_dim, 'decoder_output',
                                            bias_init_value=bias_init_values[0]))
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    self.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'decoder_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                decoder_output = tf.nn.sigmoid(
                    self.create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            self.input_dim,
                                            'decoder_output', bias_init_value=bias_init_values[1]))
                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    self.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                            'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = tf.nn.relu(
                        self.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                                bias_init_value=bias_init_values[i]))
                decoder_output = tf.nn.sigmoid(
                    self.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            self.input_dim,
                                            'decoder_output', bias_init_value=bias_init_values[-1]))
                return decoder_output

    def discriminator(self, X, bias_init_values, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param X: tensor of shape [batch_size, z_dim]
        :param bias_init_values: the initial value for the bias
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """

        # number of hidden layers
        n__hidden_layers = len(self.n_neurons_of_hidden_layer_x_discriminator)

        assert n__hidden_layers == len(bias_init_values) - 1

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Discriminator'
        with tf.name_scope('Discriminator'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = self.create_dense_layer(X, self.z_dim, 1, 'discriminator_output',
                                                               bias_init_value=bias_init_values[0])
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    self.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                discriminator_output = self.create_dense_layer(dense_layer_1,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                                               'discriminator_output',
                                                               bias_init_value=bias_init_values[1])
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    self.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        self.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                                self.n_neurons_of_hidden_layer_x_discriminator[i],
                                                'discriminator_dense_layer_' +
                                                str(i + 1),
                                                bias_init_value=bias_init_values[i]))
                discriminator_output = self.create_dense_layer(dense_layer_i,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                                               'discriminator_output',
                                                               bias_init_value=bias_init_values[-1])
                return discriminator_output

    def create_tensorboard_summary(self, decoder_output, encoder_output, autoencoder_loss, discriminator_loss,
                                   generator_loss, real_distribution):
        """
        defines what should be shown in the tensorboard summary
        :param decoder_output:
        :param encoder_output:
        :param autoencoder_loss:
        :param discriminator_loss:
        :param generator_loss:
        :param real_distribution:
        :return:
        """
        # Reshape immages to display them
        input_images = tf.reshape(self.X, [-1, 28, 28, 1])
        generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Discriminator Loss', tensor=discriminator_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
        summary_op = tf.summary.merge_all()
        return summary_op

    def form_results(self):
        """
        Forms folders for each run to store the tensorboard files, saved models and the log files.
        :return: three strings pointing to tensorboard, saved models and log paths respectively.
        """
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"). \
            replace(" ", ":").replace(":", "_")

        folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_{6}_Adversarial_Autoencoder". \
            format(date, self.loss_function_discriminator, self.z_dim, self.learning_rate_autoencoder, self.batch_size,
                   self.n_epochs, self.beta1_autoencoder)
        tensorboard_path = self.results_path + folder_name + '/Tensorboard'
        saved_model_path = self.results_path + folder_name + '/Saved_models/'
        log_path = self.results_path + folder_name + '/log'
        if not os.path.exists(self.results_path + folder_name):
            os.mkdir(self.results_path + folder_name)
            os.mkdir(tensorboard_path)
            os.mkdir(saved_model_path)
            os.mkdir(log_path)
        return tensorboard_path, saved_model_path, log_path

    def generate_image_grid(self, sess, op):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :return: None, displays a matplotlib window with all the merged images.
        """

        # TODO: draw with function, not hard coded
        # creates evenly spaced values within [-10, 10] with a spacing of 1.5
        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            # create a data point from the x_points and y_points array as input for the decoder
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z})
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.show()

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        # Get the MNIST data
        # todo: no more hard coded input
        mnist = input_data.read_data_sets('./data', one_hot=True)

        # options = dict(dtype=dtype, reshape=reshape, seed=seed)
        #
        # train = DataSet(train_images, train_labels, **options)
        # validation = DataSet(validation_images, validation_labels, **options)
        # test = DataSet(test_images, test_labels, **options)
        #
        # dataSet = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        # data = dataSet(train=train, validation=validation, test=test)

        # Saving the model
        saver = tf.train.Saver()
        step = 0
        with tf.Session() as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:
                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = self.form_results()
                writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # we want n_epochs iterations
                for i in range(self.n_epochs):

                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(mnist.train.num_examples / self.batch_size)
                    print("------------------Epoch {}/{}------------------".format(i, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        # todo: general class for real distributions:
                        #   - inherit for flower (paper p. 6)
                        #   - inherit for swiss roll  (paper p. 6)
                        z_real_dist = np.random.randn(self.batch_size, self.z_dim) * 5.

                        # get the batch from the training data
                        batch_x, batch_labels = mnist.train.next_batch(self.batch_size)
                        # print(batch_x)

                        """
                        Reconstruction phase: the autoencoder updates the encoder and the decoder to minimize the
                        reconstruction error of the inputs
                        """
                        # train the autoencoder by minimizing the reconstruction error between X and X_target
                        sess.run(self.autoencoder_optimizer, feed_dict={self.X: batch_x, self.X_target: batch_x})

                        """
                        Regularization phase: the adversarial network first updates its discriminative network
                        to tell apart the true samples (generated using the prior) from the generated samples (the 
                        hidden codes computed by the autoencoder). The adversarial network then updates its generator 
                        (which is also the encoder of the autoencoder) to confuse the discriminative network.
                        """
                        # train the discriminator to distinguish the true samples from the fake samples generated by the
                        # generator
                        sess.run(self.discriminator_optimizer,
                                 feed_dict={self.X: batch_x, self.X_target: batch_x,
                                            self.real_distribution: z_real_dist})
                        # train the generator to fool the discriminator with its generated samples.
                        sess.run(self.generator_optimizer, feed_dict={self.X: batch_x, self.X_target: batch_x})

                        # every 50 steps: write a summary
                        if b % 50 == 0:
                            a_loss, d_loss, g_loss, summary, discriminator_neg_samples_, discriminator_pos_samples_ = sess.run(
                                [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                 self.tensorboard_summary, self.discriminator_neg_samples_,
                                 self.discriminator_pos_samples_],
                                feed_dict={self.X: batch_x, self.X_target: batch_x,
                                           self.real_distribution: z_real_dist})
                            writer.add_summary(summary, global_step=step)
                            print("Epoch: {}, iteration: {}".format(i, b))
                            print("summed losses:", a_loss + d_loss + g_loss)
                            # print(discriminator_neg_samples_)
                            # print(discriminator_pos_samples_)
                            print("Autoencoder Loss: {}".format(a_loss))
                            print("Discriminator Loss: {}".format(d_loss))
                            print("Generator Loss: {}".format(g_loss))
                            with open(log_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(i, b))
                                log.write("Autoencoder Loss: {}\n".format(a_loss))
                                log.write("Discriminator Loss: {}\n".format(d_loss))
                                log.write("Generator Loss: {}\n".format(g_loss))
                        step += 1

                        # saver.save(sess, save_path=saved_model_path, global_step=step)

            # display the generated images of the latest trained autoencoder
            else:
                # Get the latest results folder
                all_results = os.listdir(self.results_path)
                all_results.sort()
                saver.restore(sess, save_path=tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1]
                                                                         + '/Saved_models/'))
                self.generate_image_grid(sess, op=self.decoder_output)
