"""
    Implementation of a unsupervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""
import json

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin
import AdversarialAutoencoderHelperFunctions


class AdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        self.result_folder_name = None
        self.parameter_dictionary = parameter_dictionary
        self.verbose = parameter_dictionary["verbose"]

        self.input_dim_x = parameter_dictionary["input_dim_x"]
        self.input_dim_y = parameter_dictionary["input_dim_y"]

        # input is RGB image
        if parameter_dictionary["color_scale"] == "rgb_scale":
            self.color_scale = "rgb_scale"
            self.input_dim = parameter_dictionary["input_dim_x"] * parameter_dictionary["input_dim_y"] * 3
        # input is gray scale image
        else:
            self.color_scale = "gray_scale"
            self.input_dim = parameter_dictionary["input_dim_x"] * parameter_dictionary["input_dim_y"]

        self.performance = None
        optimizer_autoencoder = parameter_dictionary["optimizer_autoencoder"]
        optimizer_discriminator = parameter_dictionary["optimizer_discriminator"]
        optimizer_generator = parameter_dictionary["optimizer_generator"]

        # dataset selected by the user (MNIST, SVHN, cifar10, custom)
        self.selected_dataset = parameter_dictionary["selected_dataset"]

        """
        params for network topology
        """

        # number of neurons of the hidden layers
        self.n_neurons_of_hidden_layer_x_autoencoder = parameter_dictionary["n_neurons_of_hidden_layer_x_autoencoder"]
        self.n_neurons_of_hidden_layer_x_discriminator = parameter_dictionary[
            "n_neurons_of_hidden_layer_x_discriminator"]

        # initial bias values of the hidden layers
        # TODO: remove this
        self.bias_init_value_of_hidden_layer_x_autoencoder \
            = [0.0] * (len(self.n_neurons_of_hidden_layer_x_autoencoder)+1)
        self.bias_init_value_of_hidden_layer_x_discriminator \
            = [0.0] * (len(self.n_neurons_of_hidden_layer_x_discriminator)+1)

        # self.bias_init_value_of_hidden_layer_x_autoencoder = parameter_dictionary[
        #     "bias_init_value_of_hidden_layer_x_autoencoder"]
        # self.bias_init_value_of_hidden_layer_x_discriminator = parameter_dictionary[
        #     "bias_init_value_of_hidden_layer_x_discriminator"]

        """
        params for learning
        """

        # number of epochs for training
        self.n_epochs = parameter_dictionary["n_epochs"]

        # number of training examples in one forward/backward pass
        self.batch_size = parameter_dictionary["batch_size"]

        # dimension of the latent representation
        self.z_dim = parameter_dictionary["z_dim"]

        # learning rate for the different parts of the network
        self.learning_rate_autoencoder = parameter_dictionary["learning_rate_autoencoder"]
        self.learning_rate_discriminator = parameter_dictionary["learning_rate_discriminator"]
        self.learning_rate_generator = parameter_dictionary["learning_rate_generator"]

        """
        params for optimizers
        """
        self.AdadeltaOptimizer_rho_autoencoder = parameter_dictionary["AdadeltaOptimizer_rho_autoencoder"]
        self.AdadeltaOptimizer_epsilon_autoencoder = parameter_dictionary["AdadeltaOptimizer_epsilon_autoencoder"]

        self.AdadeltaOptimizer_rho_discriminator = parameter_dictionary["AdadeltaOptimizer_rho_discriminator"]
        self.AdadeltaOptimizer_epsilon_discriminator = parameter_dictionary["AdadeltaOptimizer_epsilon_discriminator"]

        self.AdadeltaOptimizer_rho_generator = parameter_dictionary["AdadeltaOptimizer_rho_generator"]
        self.AdadeltaOptimizer_epsilon_generator = parameter_dictionary["AdadeltaOptimizer_epsilon_generator"]

        self.AdagradOptimizer_initial_accumulator_value_autoencoder = \
            parameter_dictionary["AdagradOptimizer_initial_accumulator_value_autoencoder"]

        self.AdagradOptimizer_initial_accumulator_value_discriminator = \
            parameter_dictionary["AdagradOptimizer_initial_accumulator_value_discriminator"]

        self.AdagradOptimizer_initial_accumulator_value_generator = \
            parameter_dictionary["AdagradOptimizer_initial_accumulator_value_generator"]

        self.MomentumOptimizer_momentum_autoencoder = parameter_dictionary["MomentumOptimizer_momentum_autoencoder"]
        self.MomentumOptimizer_use_nesterov_autoencoder = parameter_dictionary[
            "MomentumOptimizer_use_nesterov_autoencoder"]

        self.MomentumOptimizer_momentum_discriminator = parameter_dictionary["MomentumOptimizer_momentum_discriminator"]
        self.MomentumOptimizer_use_nesterov_discriminator = parameter_dictionary[
            "MomentumOptimizer_use_nesterov_discriminator"]

        self.MomentumOptimizer_momentum_generator = parameter_dictionary["MomentumOptimizer_momentum_generator"]
        self.MomentumOptimizer_use_nesterov_generator = parameter_dictionary["MomentumOptimizer_use_nesterov_generator"]

        self.AdamOptimizer_beta1_autoencoder = parameter_dictionary["AdamOptimizer_beta1_autoencoder"]
        self.AdamOptimizer_beta2_autoencoder = parameter_dictionary["AdamOptimizer_beta2_autoencoder"]
        self.AdamOptimizer_epsilon_autoencoder = parameter_dictionary["AdamOptimizer_epsilon_autoencoder"]

        self.AdamOptimizer_beta1_discriminator = parameter_dictionary["AdamOptimizer_beta1_discriminator"]
        self.AdamOptimizer_beta2_discriminator = parameter_dictionary["AdamOptimizer_beta2_discriminator"]
        self.AdamOptimizer_epsilon_discriminator = parameter_dictionary["AdamOptimizer_epsilon_discriminator"]

        self.AdamOptimizer_beta1_generator = parameter_dictionary["AdamOptimizer_beta1_generator"]
        self.AdamOptimizer_beta2_generator = parameter_dictionary["AdamOptimizer_beta2_generator"]
        self.AdamOptimizer_epsilon_generator = parameter_dictionary["AdamOptimizer_epsilon_generator"]

        self.FtrlOptimizer_learning_rate_power_autoencoder = \
            parameter_dictionary["FtrlOptimizer_learning_rate_power_autoencoder"]
        self.FtrlOptimizer_initial_accumulator_value_autoencoder = \
            parameter_dictionary["FtrlOptimizer_initial_accumulator_value_autoencoder"]
        self.FtrlOptimizer_l1_regularization_strength_autoencoder = \
            parameter_dictionary["FtrlOptimizer_l1_regularization_strength_autoencoder"]
        self.FtrlOptimizer_l2_regularization_strength_autoencoder = \
            parameter_dictionary["FtrlOptimizer_l2_regularization_strength_autoencoder"]
        self.FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder = \
            parameter_dictionary["FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder"]

        self.FtrlOptimizer_learning_rate_power_discriminator = \
            parameter_dictionary["FtrlOptimizer_learning_rate_power_discriminator"]
        self.FtrlOptimizer_initial_accumulator_value_discriminator = \
            parameter_dictionary["FtrlOptimizer_initial_accumulator_value_discriminator"]
        self.FtrlOptimizer_l1_regularization_strength_discriminator = \
            parameter_dictionary["FtrlOptimizer_l1_regularization_strength_discriminator"]
        self.FtrlOptimizer_l2_regularization_strength_discriminator = \
            parameter_dictionary["FtrlOptimizer_l2_regularization_strength_discriminator"]
        self.FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator = \
            parameter_dictionary["FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator"]

        self.FtrlOptimizer_learning_rate_power_generator = \
            parameter_dictionary["FtrlOptimizer_learning_rate_power_generator"]
        self.FtrlOptimizer_initial_accumulator_value_generator = \
            parameter_dictionary["FtrlOptimizer_initial_accumulator_value_generator"]
        self.FtrlOptimizer_l1_regularization_strength_generator = \
            parameter_dictionary["FtrlOptimizer_l1_regularization_strength_generator"]
        self.FtrlOptimizer_l2_regularization_strength_generator = \
            parameter_dictionary["FtrlOptimizer_l2_regularization_strength_generator"]
        self.FtrlOptimizer_l2_shrinkage_regularization_strength_generator = \
            parameter_dictionary["FtrlOptimizer_l2_shrinkage_regularization_strength_generator"]

        self.ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder"]
        self.ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder"]

        self.ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator"]
        self.ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator"]

        self.ProximalGradientDescentOptimizer_l1_regularization_strength_generator = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l1_regularization_strength_generator"]
        self.ProximalGradientDescentOptimizer_l2_regularization_strength_generator = \
            parameter_dictionary["ProximalGradientDescentOptimizer_l2_regularization_strength_generator"]

        self.ProximalAdagradOptimizer_initial_accumulator_value_autoencoder = \
            parameter_dictionary["ProximalAdagradOptimizer_initial_accumulator_value_autoencoder"]
        self.ProximalAdagradOptimizer_l1_regularization_strength_autoencoder = \
            parameter_dictionary["ProximalAdagradOptimizer_l1_regularization_strength_autoencoder"]
        self.ProximalAdagradOptimizer_l2_regularization_strength_autoencoder = \
            parameter_dictionary["ProximalAdagradOptimizer_l2_regularization_strength_autoencoder"]

        self.ProximalAdagradOptimizer_initial_accumulator_value_discriminator = \
            parameter_dictionary["ProximalAdagradOptimizer_initial_accumulator_value_discriminator"]
        self.ProximalAdagradOptimizer_l1_regularization_strength_discriminator = \
            parameter_dictionary["ProximalAdagradOptimizer_l1_regularization_strength_discriminator"]
        self.ProximalAdagradOptimizer_l2_regularization_strength_discriminator = \
            parameter_dictionary["ProximalAdagradOptimizer_l2_regularization_strength_discriminator"]

        self.ProximalAdagradOptimizer_initial_accumulator_value_generator = \
            parameter_dictionary["ProximalAdagradOptimizer_initial_accumulator_value_generator"]
        self.ProximalAdagradOptimizer_l1_regularization_strength_generator = \
            parameter_dictionary["ProximalAdagradOptimizer_l1_regularization_strength_generator"]
        self.ProximalAdagradOptimizer_l2_regularization_strength_generator = \
            parameter_dictionary["ProximalAdagradOptimizer_l2_regularization_strength_generator"]

        self.RMSPropOptimizer_decay_autoencoder = parameter_dictionary["RMSPropOptimizer_decay_autoencoder"]
        self.RMSPropOptimizer_momentum_autoencoder = parameter_dictionary["RMSPropOptimizer_momentum_autoencoder"]
        self.RMSPropOptimizer_epsilon_autoencoder = parameter_dictionary["RMSPropOptimizer_epsilon_autoencoder"]
        self.RMSPropOptimizer_centered_autoencoder = parameter_dictionary["RMSPropOptimizer_centered_autoencoder"]

        self.RMSPropOptimizer_decay_discriminator = parameter_dictionary["RMSPropOptimizer_decay_discriminator"]
        self.RMSPropOptimizer_momentum_discriminator = parameter_dictionary["RMSPropOptimizer_momentum_discriminator"]
        self.RMSPropOptimizer_epsilon_discriminator = parameter_dictionary["RMSPropOptimizer_epsilon_discriminator"]
        self.RMSPropOptimizer_centered_discriminator = parameter_dictionary["RMSPropOptimizer_centered_discriminator"]

        self.RMSPropOptimizer_decay_generator = parameter_dictionary["RMSPropOptimizer_decay_generator"]
        self.RMSPropOptimizer_momentum_generator = parameter_dictionary["RMSPropOptimizer_momentum_generator"]
        self.RMSPropOptimizer_epsilon_generator = parameter_dictionary["RMSPropOptimizer_epsilon_generator"]
        self.RMSPropOptimizer_centered_generator = parameter_dictionary["RMSPropOptimizer_centered_generator"]

        # exponential decay rate for the 1st moment estimates for the adam optimizer.
        self.AdamOptimizer_beta1_autoencoder = parameter_dictionary["AdamOptimizer_beta1_autoencoder"]
        self.AdamOptimizer_beta1_discriminator = parameter_dictionary["AdamOptimizer_beta1_discriminator"]
        self.AdamOptimizer_beta1_generator = parameter_dictionary["AdamOptimizer_beta1_generator"]

        # exponential decay rate for the 2nd moment estimates for the adam optimizer.
        self.AdamOptimizer_beta2_autoencoder = parameter_dictionary["AdamOptimizer_beta2_autoencoder"]
        self.AdamOptimizer_beta2_discriminator = parameter_dictionary["AdamOptimizer_beta2_discriminator"]
        self.AdamOptimizer_beta2_generator = parameter_dictionary["AdamOptimizer_beta2_generator"]

        """
        loss function
        """

        # loss function
        self.loss_function_discriminator = parameter_dictionary["loss_function_discriminator"]
        self.loss_function_generator = parameter_dictionary["loss_function_generator"]

        # path for the results
        self.results_path = parameter_dictionary["results_path"]

        """
        placeholder variables 
        """

        # holds the input data
        self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Input')
        # holds the desired output of the autoencoder
        self.X_target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Target')
        # holds the real distribution p(z) used as positive sample for the discriminator
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim],
                                                name='Real_distribution')
        # holds the input samples for the decoder (only for generating the images; NOT used for training)
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim], name='Decoder_input')
        self.decoder_input_multiple = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim],
                                                     name='Decoder_input_multiple')

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

        # output of the decoder
        with tf.variable_scope(tf.get_variable_scope()):
            # decoder output: only used for generating the images for tensorboard
            self.decoder_output = self.decoder(self.decoder_input, reuse=True,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

            self.decoder_output_multiple = \
                self.decoder(self.decoder_input_multiple, reuse=True,
                             bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        """
        Init the loss functions
        """

        # Autoencoder loss
        # self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - decoder_output))
        self.autoencoder_loss = tf.reduce_mean(tf.abs(self.X_target - decoder_output))

        # Discriminator Loss
        discriminator_loss_pos_samples = tf.reduce_mean(
            AdversarialAutoencoderHelperFunctions.get_loss_function(loss_function=self.loss_function_discriminator,
                                                                    labels=tf.ones_like(discriminator_pos_samples),
                                                                    logits=discriminator_pos_samples))
        discriminator_loss_neg_samples = tf.reduce_mean(
            AdversarialAutoencoderHelperFunctions.get_loss_function(loss_function=self.loss_function_discriminator,
                                                                    labels=tf.zeros_like(discriminator_neg_samples),
                                                                    logits=discriminator_neg_samples))
        self.discriminator_loss = discriminator_loss_neg_samples + discriminator_loss_pos_samples

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            AdversarialAutoencoderHelperFunctions.get_loss_function(loss_function=self.loss_function_generator,
                                                                    labels=tf.ones_like(discriminator_neg_samples),
                                                                    logits=discriminator_neg_samples))

        """
        Init the optimizers
        """

        all_variables = tf.trainable_variables()
        discriminator_vars = [var for var in all_variables if 'discriminator_' in var.name]
        encoder_vars = [var for var in all_variables if 'encoder_' in var.name]

        # Optimizers
        self.autoencoder_optimizer = AdversarialAutoencoderHelperFunctions. \
            get_optimizer_autoencoder(self, optimizer_autoencoder).minimize(self.autoencoder_loss)
        self.discriminator_optimizer = AdversarialAutoencoderHelperFunctions. \
            get_optimizer_discriminator(self, optimizer_discriminator).minimize(self.discriminator_loss,
                                                                                var_list=discriminator_vars)
        self.generator_optimizer = AdversarialAutoencoderHelperFunctions. \
            get_optimizer_generator(self, optimizer_generator).minimize(self.generator_loss,
                                                                        var_list=encoder_vars)

        """
        Create the tensorboard summary
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=decoder_output, encoder_output=encoder_output,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_loss=self.discriminator_loss,
                                            generator_loss=self.generator_loss,
                                            real_distribution=self.real_distribution,
                                            decoder_output_multiple=self.decoder_output_multiple)

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    def get_performance(self):
        return self.performance

    def get_result_folder_name(self):
        return self.result_folder_name

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
            if n_hidden_layers == 0:
                latent_variable = AdversarialAutoencoderHelperFunctions. \
                    create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                       bias_init_value=bias_init_values[0])
                return latent_variable
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(AdversarialAutoencoderHelperFunctions.
                                           create_dense_layer(X, self.input_dim,
                                                              self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                              'encoder_dense_layer_1',
                                                              bias_init_value=bias_init_values[0]))
                latent_variable = AdversarialAutoencoderHelperFunctions. \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0], self.z_dim,
                                       'encoder_output', bias_init_value=bias_init_values[1])
                return latent_variable
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    AdversarialAutoencoderHelperFunctions.
                        create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n_hidden_layers):
                    dense_layer_i = \
                        tf.nn.relu(AdversarialAutoencoderHelperFunctions.
                                   create_dense_layer(dense_layer_i,
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                      'encoder_dense_layer_' + str(i + 1),
                                                      bias_init_value=bias_init_values[i]))
                latent_variable = AdversarialAutoencoderHelperFunctions. \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                       self.z_dim, 'encoder_output', bias_init_value=bias_init_values[-1])
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
                    AdversarialAutoencoderHelperFunctions.create_dense_layer(X, self.z_dim, self.input_dim,
                                                                             'decoder_output',
                                                                             bias_init_value=bias_init_values[0]))
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(AdversarialAutoencoderHelperFunctions.
                                           create_dense_layer(X, self.z_dim,
                                                              self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                              'decoder_dense_layer_1',
                                                              bias_init_value=bias_init_values[0]))
                decoder_output = tf.nn.sigmoid(
                    AdversarialAutoencoderHelperFunctions.
                        create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           self.input_dim, 'decoder_output', bias_init_value=bias_init_values[1]))
                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    AdversarialAutoencoderHelperFunctions.
                        create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                           'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = tf.nn.relu(
                        AdversarialAutoencoderHelperFunctions.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                               self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                               'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                               bias_init_value=bias_init_values[i]))
                decoder_output = tf.nn.sigmoid(
                    AdversarialAutoencoderHelperFunctions.
                        create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           self.input_dim, 'decoder_output', bias_init_value=bias_init_values[-1]))
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
                discriminator_output = AdversarialAutoencoderHelperFunctions.\
                    create_dense_layer(X, self.z_dim, 1, 'discriminator_output', bias_init_value=bias_init_values[0])
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    AdversarialAutoencoderHelperFunctions
                        .create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                discriminator_output = AdversarialAutoencoderHelperFunctions\
                    .create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                        'discriminator_output', bias_init_value=bias_init_values[1])
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    AdversarialAutoencoderHelperFunctions
                        .create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        AdversarialAutoencoderHelperFunctions.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                               self.n_neurons_of_hidden_layer_x_discriminator[i],
                                               'discriminator_dense_layer_' + str(i + 1),
                                               bias_init_value=bias_init_values[i]))

                discriminator_output = AdversarialAutoencoderHelperFunctions.\
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                       'discriminator_output',  bias_init_value=bias_init_values[-1])
                return discriminator_output

    def create_tensorboard_summary(self, decoder_output, encoder_output, autoencoder_loss, discriminator_loss,
                                   generator_loss, real_distribution, decoder_output_multiple):
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

        # Reshape images accordingly to the color scale to display them
        if self.color_scale == "rgb_scale":
            input_images = AdversarialAutoencoderHelperFunctions.reshape_to_rgb_image(self.X, self.input_dim_x,
                                                                                      self.input_dim_y)
            generated_images = AdversarialAutoencoderHelperFunctions.reshape_to_rgb_image(decoder_output,
                                                                                          self.input_dim_x,
                                                                                          self.input_dim_y)
            generated_images_z_dist = AdversarialAutoencoderHelperFunctions.reshape_to_rgb_image(
                decoder_output_multiple, self.input_dim_x,
                self.input_dim_y)
        else:
            input_images = tf.reshape(self.X, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images = tf.reshape(decoder_output, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images_z_dist = tf.reshape(decoder_output_multiple, [-1, self.input_dim_x, self.input_dim_y, 1])

        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Discriminator Loss', tensor=discriminator_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=50)
        tf.summary.image(name='Generated Images from Input Images', tensor=generated_images, max_outputs=50)
        tf.summary.image(name='Generated Images z-dist', tensor=generated_images_z_dist, max_outputs=50)
        summary_op = tf.summary.merge_all()
        return summary_op

    def generate_image_grid(self, sess, op, epoch):
        """
        Generates and saves a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :return:
        """

        # TODO: draw with function, not hard coded
        # creates evenly spaced values within [-10, 10] with a spacing of 1.5
        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        # create the image grid
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        # iterate over the image grid
        for i, g in enumerate(gs):

            # create a data point from the x_points and y_points array as input for the decoder
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z})
            ax = plt.subplot(g)

            # reshape the images according to the color scale
            if self.color_scale == "gray_scale":
                # matplotlib wants a 2D array
                img = np.array(x.tolist()).reshape(self.input_dim_x, self.input_dim_y)
                # show the image
                ax.imshow(img, cmap='gray')
            else:
                image_array = np.array(x.tolist())
                n_colored_pixels_per_channel = self.input_dim_x * self.input_dim_y
                # first n_colored_pixels_per_channel encode red
                red_pixels = image_array[:, :n_colored_pixels_per_channel].reshape(self.input_dim_x,
                                                                                   self.input_dim_y, 1)
                # next n_colored_pixels_per_channel encode green
                green_pixels = image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                    .reshape(self.input_dim_x, self.input_dim_y, 1)
                # last n_colored_pixels_per_channel encode blue
                blue_pixels = image_array[:, n_colored_pixels_per_channel * 2:].reshape(self.input_dim_x,
                                                                                        self.input_dim_y, 1)
                # concatenate the color arrays into one array
                img = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)
                # show the image
                ax.imshow(img)

            # we don't want ticks for the x or the y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            # create the label for the y axis
            if ax.is_first_col():
                ax.set_ylabel(x_points[int(i / ny)], fontsize=9)

            # create the label x for the x axis
            if ax.is_last_row():
                ax.set_xlabel(y_points[int(i % ny)], fontsize=9)

        # save the created image grid
        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

    def generate_image_grid_z_dim(self, sess, op, epoch, image_grid_shape=(10, 10)):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param image_grid_shape: shape of the resulting image grid
        :return: None, displays a matplotlib window with all the merged images.
        """

        image_grid_x_length = image_grid_shape[0]
        image_grid_y_length = image_grid_shape[1]
        n_points_to_sample = image_grid_x_length * image_grid_y_length

        # randomly sample some points from the z dim space
        # TODO: draw with function, not hard coded
        random_points = np.random.uniform(-10, 10, [n_points_to_sample, self.z_dim])

        # this is working (in train function) as input
        # random_points = np.random.uniform(-10, 10, [self.batch_size, self.z_dim])

        # print(random_points)

        plt.subplot()
        gs = gridspec.GridSpec(image_grid_x_length, image_grid_y_length, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            # create a data point from the x_points and y_points array as input for the decoder
            z = random_points[i]
            z = np.reshape(z, (1, self.z_dim))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z})
            ax = plt.subplot(g)

            # reshape the images according to the color scale
            if self.color_scale == "gray_scale":
                # matplotlib wants a 2D array
                img = np.array(x.tolist()).reshape(self.input_dim_x, self.input_dim_y)
                # show the image
                ax.imshow(img, cmap='gray')
            else:
                image_array = np.array(x.tolist())
                n_colored_pixels_per_channel = self.input_dim_x * self.input_dim_y
                # first n_colored_pixels_per_channel encode red
                red_pixels = image_array[:, :n_colored_pixels_per_channel].reshape(self.input_dim_x,
                                                                                   self.input_dim_y, 1)
                # next n_colored_pixels_per_channel encode green
                green_pixels = image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                    .reshape(self.input_dim_x, self.input_dim_y, 1)
                # last n_colored_pixels_per_channel encode blue
                blue_pixels = image_array[:, n_colored_pixels_per_channel * 2:].reshape(self.input_dim_x,
                                                                                        self.input_dim_y, 1)
                # concatenate the color arrays into one array
                img = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)
                # show the image
                ax.imshow(img)

            # we don't want ticks for the x or the y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        log_path = None

        # Get the data
        data = AdversarialAutoencoderHelperFunctions.get_input_data(self.selected_dataset)

        # Saving the model
        saver = tf.train.Saver()
        # TODO: maybe worth a try..
        # saver = tf.train.Saver(tf.trainable_variables())

        autoencoder_loss_final, discriminator_loss_final, generator_loss_final = 0, 0, 0

        step = 0
        with tf.Session() as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:
                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = AdversarialAutoencoderHelperFunctions.form_results(self)
                writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # we want n_epochs iterations
                for i in range(self.n_epochs):

                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(data.train.num_examples / self.batch_size)

                    if self.verbose:
                        print("------------------Epoch {}/{}------------------".format(i, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        # todo: general class for real distributions:
                        #   - inherit for flower (paper p. 6)
                        #   - inherit for swiss roll  (paper p. 6)
                        z_real_dist = np.random.randn(self.batch_size, self.z_dim) * 5.
                        # z_real_dist = np.random.normal(scale=5., size=[self.batch_size, self.z_dim])

                        # get the batch from the training data
                        batch_x, batch_labels = data.train.next_batch(self.batch_size)
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

                        # every 5 epochs: write a summary
                        if i % 5 == 0:

                            # TODO: draw from function not hard coded
                            # draw some random points as input for the decoder
                            random_points = np.random.uniform(-10, 10, [self.batch_size, self.z_dim])

                            a_loss, d_loss, g_loss, summary = sess.run(
                                [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                 self.tensorboard_summary],
                                feed_dict={self.X: batch_x, self.X_target: batch_x,
                                           self.real_distribution: z_real_dist,
                                           self.decoder_input_multiple: random_points})
                            writer.add_summary(summary, global_step=step)

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(i, b))
                                print("summed losses:", a_loss + d_loss + g_loss)
                                print("Autoencoder Loss: {}".format(a_loss))
                                print("Discriminator Loss: {}".format(d_loss))
                                print("Generator Loss: {}".format(g_loss))

                            autoencoder_loss_final = a_loss
                            discriminator_loss_final = d_loss
                            generator_loss_final = g_loss

                            with open(log_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(i, b))
                                log.write("Autoencoder Loss: {}\n".format(a_loss))
                                log.write("Discriminator Loss: {}\n".format(d_loss))
                                log.write("Generator Loss: {}\n".format(g_loss))
                        step += 1

                    if i % 25 == 0:
                        # TODO: maybe two separate functions are not needed..
                        if self.z_dim > 2:
                            self.generate_image_grid_z_dim(sess, op=self.decoder_output, epoch=i)
                        else:
                            self.generate_image_grid(sess, op=self.decoder_output, epoch=i)

                            # saver.save(sess, save_path=saved_model_path, global_step=step)

            # display the generated images of the latest trained autoencoder
            else:
                # Get the latest results folder
                all_results = os.listdir(self.results_path)
                all_results.sort()

                saver.restore(sess, save_path=tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1]
                                                                         + '/Saved_models/'))
                # self.generate_image_grid(sess, op=self.decoder_output)

        # write the parameter dictionary to some file
        json_dictionary = json.dumps(self.parameter_dictionary)
        with open(log_path + '/params.txt', 'a') as file:
            file.write(json_dictionary)

        if self.verbose:
            print("Autoencoder Loss: {}".format(autoencoder_loss_final))
            print("Discriminator Loss: {}".format(discriminator_loss_final))
            print("Generator Loss: {}".format(generator_loss_final))

        # TODO: use weights for the losses (autoencoder loss more important than discrimininator..)
        self.performance = {"autoencoder_loss_final": autoencoder_loss_final,
                            "discriminator_loss_final": discriminator_loss_final,
                            "generator_loss_final": generator_loss_final,
                            "summed_loss_final": autoencoder_loss_final + discriminator_loss_final + generator_loss_final}

        sess.close()
        tf.reset_default_graph()
