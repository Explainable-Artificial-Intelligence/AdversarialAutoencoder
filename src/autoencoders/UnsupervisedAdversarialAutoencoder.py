"""
    Implementation of a unsupervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""
import glob
import json

import imageio
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin
import re

import src.util.AdversarialAutoencoderHelperFunctions as aae_helper
from src.util.Distributions import draw_from_multiple_gaussians, draw_from_single_gaussian, draw_from_swiss_roll


class AdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        self.train_status = "start"
        self.result_folder_name = None
        self.parameter_dictionary = parameter_dictionary
        self.verbose = parameter_dictionary["verbose"]
        self.save_final_model = parameter_dictionary["save_final_model"]    # whether to save the final model

        """
        params for the data 
        """

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
        self.bias_init_value_of_hidden_layer_x_autoencoder = parameter_dictionary[
            "bias_init_value_of_hidden_layer_x_autoencoder"]
        self.bias_init_value_of_hidden_layer_x_discriminator = parameter_dictionary[
            "bias_init_value_of_hidden_layer_x_discriminator"]

        # activation functions for the different parts of the network
        self.activation_function_encoder = parameter_dictionary["activation_function_encoder"]
        self.activation_function_decoder = parameter_dictionary["activation_function_decoder"]
        self.activation_function_discriminator = parameter_dictionary["activation_function_discriminator"]

        """
        params for learning
        """

        # number of epochs for training
        self.n_epochs = parameter_dictionary["n_epochs"]

        # number of training examples in one forward/backward pass
        self.batch_size = parameter_dictionary["batch_size"]

        # dimension of the latent representation
        self.z_dim = parameter_dictionary["z_dim"]

        # Create a variable to track the global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # learning rate for the different parts of the network
        self.learning_rate_autoencoder = parameter_dictionary["learning_rate_autoencoder"]
        self.learning_rate_discriminator = parameter_dictionary["learning_rate_discriminator"]
        self.learning_rate_generator = parameter_dictionary["learning_rate_generator"]

        # learning rate for the different parts of the network
        self.decaying_learning_rate_name_autoencoder = parameter_dictionary["decaying_learning_rate_name_autoencoder"]
        self.decaying_learning_rate_name_discriminator = \
            parameter_dictionary["decaying_learning_rate_name_discriminator"]
        self.decaying_learning_rate_name_generator = parameter_dictionary["decaying_learning_rate_name_generator"]

        """
        params for optimizers
        """
        if True:

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
            self.encoder_output = self.encoder(self.X,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # decoder part of the autoencoder
            self.decoder_output = self.decoder(self.encoder_output,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        # init discriminator
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive samples p(z) (from a real data distribution)
            self.discriminator_pos_samples = \
                self.discriminator(self.real_distribution,
                                   bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)
            # discriminator for the negative samples q(z) (generated by the generator)
            self.discriminator_neg_samples = \
                self.discriminator(self.encoder_output, reuse=True,
                                   bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)

        # output of the decoder
        with tf.variable_scope(tf.get_variable_scope()):
            # used for "manually" passing single values drawn from a real distribution through the decoder for the
            # tensorboard
            self.decoder_output_real_dist = self.decoder(self.decoder_input, reuse=True,
                                                         bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # used for "manually" passing multiple values drawn from a real distribution through the decoder for the
            # tensorboard
            self.decoder_output_multiple = \
                self.decoder(self.decoder_input_multiple, reuse=True,
                             bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        """
        Init the loss functions
        """

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - self.decoder_output))

        # Discriminator Loss
        discriminator_loss_pos_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.ones_like(self.discriminator_pos_samples),
                                         logits=self.discriminator_pos_samples))
        discriminator_loss_neg_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.zeros_like(self.discriminator_neg_samples),
                                         logits=self.discriminator_neg_samples))
        self.discriminator_loss = discriminator_loss_neg_samples + discriminator_loss_pos_samples

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_generator,
                                         labels=tf.ones_like(self.discriminator_neg_samples),
                                         logits=self.discriminator_neg_samples))

        """
        Init the optimizers
        """

        optimizer_autoencoder = parameter_dictionary["optimizer_autoencoder"]
        optimizer_discriminator = parameter_dictionary["optimizer_discriminator"]
        optimizer_generator = parameter_dictionary["optimizer_generator"]

        # get the discriminator and encoder variables
        all_variables = tf.trainable_variables()
        discriminator_vars = [var for var in all_variables if 'discriminator_' in var.name]
        encoder_vars = [var for var in all_variables if 'encoder_' in var.name]

        # Optimizers
        self.autoencoder_optimizer = aae_helper. \
            get_optimizer(self, optimizer_autoencoder, "autoencoder", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.autoencoder_trainer = self.autoencoder_optimizer.minimize(self.autoencoder_loss)
        self.discriminator_optimizer = aae_helper. \
            get_optimizer(self, optimizer_discriminator, "discriminator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator)
        self.discriminator_trainer = self.discriminator_optimizer.minimize(self.discriminator_loss,
                                                                           var_list=discriminator_vars)
        self.generator_optimizer = aae_helper. \
            get_optimizer(self, optimizer_generator, "generator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_generator)
        self.generator_trainer = self.generator_optimizer.minimize(self.generator_loss, var_list=encoder_vars,
                                                                   global_step=self.global_step)

        """
        Create the tensorboard summary
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=self.decoder_output, encoder_output=self.encoder_output,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_loss=self.discriminator_loss,
                                            generator_loss=self.generator_loss,
                                            real_distribution=self.real_distribution,
                                            decoder_output_multiple=self.decoder_output_multiple)

        """
        Variable for the "manual" summary
        """
        self.final_performance = None
        self.performance_over_time = {"autoencoder_losses": [], "discriminator_losses": [], "generator_losses": [],
                                      "list_of_epochs": []}
        self.learning_rates = {"autoencoder_lr": [], "discriminator_lr": [], "generator_lr": [], "list_of_epochs": []}

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    def get_train_status(self):
        return self.train_status

    def set_train_status(self, status_to_set):
        self.train_status = status_to_set

    def get_final_performance(self):
        return self.final_performance

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
                latent_variable = aae_helper. \
                    create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                       bias_init_value=bias_init_values[0])
                return latent_variable
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_encoder, aae_helper.
                                           create_dense_layer(X, self.input_dim,
                                                              self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                              'encoder_dense_layer_1',
                                                              bias_init_value=bias_init_values[0]))
                latent_variable = aae_helper. \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0], self.z_dim,
                                       'encoder_output', bias_init_value=bias_init_values[1])
                return latent_variable
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_encoder,
                    aae_helper.
                        create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n_hidden_layers):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_encoder, aae_helper.
                                                                                 create_dense_layer(dense_layer_i,
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                      'encoder_dense_layer_' + str(i + 1),
                                                      bias_init_value=bias_init_values[i]))
                latent_variable = aae_helper. \
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
                decoder_output = aae_helper.use_activation_function_for_layer(self.activation_function_decoder,
                    aae_helper.create_dense_layer(X, self.z_dim, self.input_dim,
                                                                             'decoder_output',
                                                                             bias_init_value=bias_init_values[0]))
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_decoder, aae_helper.
                                           create_dense_layer(X, self.z_dim,
                                                              self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                              'decoder_dense_layer_1',
                                                              bias_init_value=bias_init_values[0]))
                decoder_output = tf.nn.sigmoid(
                    aae_helper.
                        create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           self.input_dim, 'decoder_output', bias_init_value=bias_init_values[1]))
                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_decoder,
                    aae_helper.
                        create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                           'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_decoder,
                        aae_helper.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                               self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                               'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                               bias_init_value=bias_init_values[i]))
                decoder_output = tf.nn.sigmoid(
                    aae_helper.
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
                discriminator_output = aae_helper.\
                    create_dense_layer(X, self.z_dim, 1, 'discriminator_output', bias_init_value=bias_init_values[0])
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator,
                                                                 aae_helper.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                discriminator_output = aae_helper\
                    .create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                        'discriminator_output', bias_init_value=bias_init_values[1])
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator,
                    aae_helper
                        .create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator,
                        aae_helper.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                               self.n_neurons_of_hidden_layer_x_discriminator[i],
                                               'discriminator_dense_layer_' + str(i + 1),
                                               bias_init_value=bias_init_values[i]))

                discriminator_output = aae_helper.\
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                       'discriminator_output',  bias_init_value=bias_init_values[-1])
                return discriminator_output

    def create_tensorboard_summary(self, decoder_output, encoder_output, autoencoder_loss, discriminator_loss,
                                   generator_loss, real_distribution, decoder_output_multiple):
        """
        defines what should be shown in the tensorboard summary
        :param decoder_output: decoder output for some input images (= reconstruction image)
        :param encoder_output: encoder distribution to see whether it was able to map to the provided real distribution
        :param autoencoder_loss: loss of the autoencoder
        :param discriminator_loss: loss of the discriminator
        :param generator_loss: loss of the generator
        :param real_distribution: used real distribution the encoder should map his output to
        :param decoder_output_multiple: output of the decoder for some points drawn from the latent space
        :return:
        """

        # Reshape input images and the decoder outputs accordingly to the color scale to display them
        if self.color_scale == "rgb_scale":
            input_images = aae_helper.reshape_tensor_to_rgb_image(self.X, self.input_dim_x, self.input_dim_y)
            generated_images = aae_helper.reshape_tensor_to_rgb_image(decoder_output, self.input_dim_x,
                                                                      self.input_dim_y)
            generated_images_z_dist = aae_helper.reshape_tensor_to_rgb_image(decoder_output_multiple,
                                                                             self.input_dim_x, self.input_dim_y)
        else:
            input_images = tf.reshape(self.X, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images = tf.reshape(decoder_output, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images_z_dist = tf.reshape(decoder_output_multiple, [-1, self.input_dim_x,
                                                                           self.input_dim_y, 1])

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

    def generate_image_grid(self, sess, op, epoch, left_cell=None):
        """
        Generates and saves a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param left_cell: left cell of the grid spec with two adjacent horizontal cells holding the image grid
        and the class distribution on the latent space; if left_cell is None, then only the image grid is supposed
        to be plotted and not the "combinated" image (image grid + class distr. on latent space).
        :return:
        """

        # creates evenly spaced values within [-10, 10] with a spacing of 1.5
        x_points = np.arange(10, -10, -1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        # create the image grid
        if left_cell:
            gs = gridspec.GridSpecFromSubplotSpec(nx, ny, left_cell)
        else:
            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        # iterate over the image grid
        for i, g in enumerate(gs):

            # create a data point from the x_points and y_points array as input for the decoder
            z = np.concatenate(([y_points[int(i % nx)]], [x_points[int(i / ny)]]))
            z = np.reshape(z, (1, 2))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z})
            x = np.array(x).reshape(self.input_dim)
            ax = plt.subplot(g)

            # reshape the image array and display it
            img = aae_helper.reshape_image_array(self, x)
            if self.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            # create the label for the y axis
            if ax.is_first_col():
                ax.set_ylabel(x_points[int(i / ny)], fontsize=9)

            # create the label x for the x axis
            if ax.is_last_row():
                ax.set_xlabel(y_points[int(i % ny)], fontsize=9)

        if not left_cell:
            # save the created image grid
            plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

    def generate_image_grid_z_dim(self, sess, op, epoch, image_grid_shape=(10, 10), left_cell=None):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param image_grid_shape: shape of the resulting image grid
        :param left_cell: left cell of the grid spec with two adjacent horizontal cells holding the image grid
        and the class distribution on the latent space; if left_cell is None, then only the image grid is supposed
        to be plotted and not the "combinated" image (image grid + class distr. on latent space).
        :return: None, displays a matplotlib window with all the merged images.
        """

        image_grid_x_length = image_grid_shape[0]
        image_grid_y_length = image_grid_shape[1]
        n_points_to_sample = image_grid_x_length * image_grid_y_length

        # randomly sample some points from the z dim space
        random_points = np.random.uniform(-10, 10, [n_points_to_sample, self.z_dim])

        # create the image grid
        if left_cell:
            gs = gridspec.GridSpecFromSubplotSpec(image_grid_x_length, image_grid_y_length, left_cell)
        else:
            plt.subplot()
            gs = gridspec.GridSpec(image_grid_x_length, image_grid_y_length, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            # create a data point from the x_points and y_points array as input for the decoder
            z = random_points[i]
            z = np.reshape(z, (1, self.z_dim))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z})
            ax = plt.subplot(g)

            # reshape the image array and display it
            img = aae_helper.reshape_image_array(self, x)
            if self.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)

            # we don't want ticks for the x or the y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

        if not left_cell:
            plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        saved_model_path = None

        latent_representations_current_epoch = []
        labels_current_epoch = []

        # Get the data
        data = aae_helper.get_input_data(self.selected_dataset)

        # Saving the model
        saver = tf.train.Saver()

        autoencoder_loss_final, discriminator_loss_final, generator_loss_final = 0, 0, 0

        step = 0
        with tf.Session() as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:

                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = aae_helper.form_results(self)
                writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # write the used parameter dictionary to some file
                json_dictionary = json.dumps(self.parameter_dictionary)
                with open(log_path + '/params.txt', 'w') as file:
                    file.write(json_dictionary)

                # we want n_epochs iterations
                for epoch in range(self.n_epochs):

                    if self.train_status == "stop":

                        self.end_training(autoencoder_loss_final, discriminator_loss_final, generator_loss_final,
                                          saved_model_path, saver, sess, step)
                        break

                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(data.train.num_examples / self.batch_size)

                    print("n training examples:", data.train.num_examples)
                    print("batch size:", self.batch_size)
                    print("n batches:", n_batches)

                    if self.verbose:
                        print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        # z_real_dist = draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(self.batch_size, self.z_dim))
                        z_real_dist = \
                            draw_from_single_gaussian(mean=0.0, std_dev=1.0, shape=(self.batch_size, self.z_dim)) * 5

                        # get the batch from the training data
                        batch_x, batch_labels = data.train.next_batch(self.batch_size)

                        """
                        Reconstruction phase: the autoencoder updates the encoder and the decoder to minimize the
                        reconstruction error of the inputs
                        """
                        # train the autoencoder by minimizing the reconstruction error between X and X_target
                        # sess.run(self.autoencoder_optimizer, feed_dict={self.X: batch_x, self.X_target: batch_x})
                        sess.run(self.autoencoder_trainer, feed_dict={self.X: batch_x, self.X_target: batch_x})

                        """
                        Regularization phase: the adversarial network first updates its discriminative network
                        to tell apart the true samples (generated using the prior) from the generated samples (the 
                        hidden codes computed by the autoencoder). The adversarial network then updates its generator 
                        (which is also the encoder of the autoencoder) to confuse the discriminative network.
                        """
                        # train the discriminator to distinguish the true samples from the fake samples generated by the
                        # generator
                        sess.run(self.discriminator_trainer,
                                 feed_dict={self.X: batch_x, self.X_target: batch_x,
                                            self.real_distribution: z_real_dist})
                        # train the generator to fool the discriminator with its generated samples.
                        sess.run(self.generator_trainer, feed_dict={self.X: batch_x, self.X_target: batch_x})

                        # every 5 epochs: write a summary for every 50th minibatch
                        if epoch % 5 == 0 and b % 50 == 0:

                            autoencoder_loss, discriminator_loss, generator_loss, summary, real_dist, \
                            latent_representation, discriminator_neg, discriminator_pos, decoder_output = \
                                sess.run(
                                [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                 self.tensorboard_summary, self.real_distribution, self.encoder_output,
                                 self.discriminator_neg_samples, self.discriminator_pos_samples,
                                 self.decoder_output],
                                feed_dict={self.X: batch_x, self.X_target: batch_x,
                                           self.real_distribution: z_real_dist,
                                           self.decoder_input_multiple: z_real_dist})
                            writer.add_summary(summary, global_step=step)

                            latent_representations_current_epoch.extend(latent_representation)
                            labels_current_epoch.extend(batch_labels)

                            # update the dictionary holding the losses
                            self.performance_over_time["autoencoder_losses"].append(autoencoder_loss)
                            self.performance_over_time["discriminator_losses"].append(discriminator_loss)
                            self.performance_over_time["generator_losses"].append(generator_loss)
                            self.performance_over_time["list_of_epochs"].append(epoch + (b / n_batches))

                            # update the dictionary holding the learning rates
                            self.learning_rates["autoencoder_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.autoencoder_optimizer)))
                            self.learning_rates["discriminator_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.discriminator_optimizer)))
                            self.learning_rates["generator_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.generator_optimizer)))
                            self.learning_rates["list_of_epochs"].append(epoch + (b / n_batches))

                            # create the summary image for the current minibatch
                            aae_helper.create_minibatch_summary_image(self, real_dist, latent_representation,
                                                                      discriminator_neg, discriminator_pos, batch_x,
                                                                      decoder_output, epoch, b, batch_labels)

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(epoch, b))
                                print("summed losses:", autoencoder_loss + discriminator_loss + generator_loss)
                                print("Autoencoder Loss: {}".format(autoencoder_loss))
                                print("Discriminator Loss: {}".format(discriminator_loss))
                                print("Generator Loss: {}".format(generator_loss))
                                print('Learning rate autoencoder: {}'.format(
                                    sess.run(aae_helper.get_learning_rate_for_optimizer(self.autoencoder_optimizer))))
                                print('Learning rate discriminator: {}'.format(
                                    sess.run(aae_helper.get_learning_rate_for_optimizer(self.discriminator_optimizer))))
                                print('Learning rate generator: {}'.format(
                                    sess.run(aae_helper.get_learning_rate_for_optimizer(self.generator_optimizer))))

                            autoencoder_loss_final = autoencoder_loss
                            discriminator_loss_final = discriminator_loss
                            generator_loss_final = generator_loss

                            with open(log_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                                log.write("Autoencoder Loss: {}\n".format(autoencoder_loss))
                                log.write("Discriminator Loss: {}\n".format(discriminator_loss))
                                log.write("Generator Loss: {}\n".format(generator_loss))

                        step += 1

                    # every 5 epochs ..
                    if epoch % 5 == 0:
                        # increase figure size
                        plt.rcParams["figure.figsize"] = (6.4*2, 4.8)
                        outer_grid = gridspec.GridSpec(1, 2)
                        left_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid

                        # generate the image grid for the latent space
                        # TODO: maybe two separate functions are not needed..
                        if self.z_dim > 2:
                            self.generate_image_grid_z_dim(sess, op=self.decoder_output_real_dist, epoch=epoch,
                                                           left_cell=left_cell)
                        else:
                            self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=epoch,
                                                     left_cell=left_cell)
                        # plt.close('all')

                        # draw the class distribution on the latent space
                        result_path = self.results_path + self.result_folder_name + '/Tensorboard/'
                        aae_helper.draw_class_distribution_on_latent_space(latent_representations_current_epoch,
                                                                           labels_current_epoch, result_path, epoch,
                                                                           combined_plot=True)

                    # reset the list holding the latent representations for the current epoch
                    latent_representations_current_epoch = []
                    labels_current_epoch = []

            # display the generated images of the latest trained autoencoder
            else:
                # Get the latest results folder
                all_results = os.listdir(self.results_path)
                all_results.sort()
                saver.restore(sess, save_path=tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1]
                                                                         + '/Saved_models/'))
                self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch="last")

            # TODO: param whether model should be saved (?)
            self.end_training(autoencoder_loss_final, discriminator_loss_final, generator_loss_final,
                              None, saver, sess, step)
            # self.end_training(autoencoder_loss_final, discriminator_loss_final, generator_loss_final,
            #                   saved_model_path, saver, sess, step)

    def end_training(self, autoencoder_loss_final, discriminator_loss_final, generator_loss_final, saved_model_path,
                     saver, sess, step):
        """
        ends the training by saving the model if a model path is provided, saving the final losses and closing the
        tf session
        :param autoencoder_loss_final: final loss of the autoencoder
        :param discriminator_loss_final: final loss of the discriminator
        :param generator_loss_final: final loss of the generator
        :param saved_model_path: path where the saved model should be stored
        :param saver: tf.train.Saver() to save the model
        :param sess: session to save and to close
        :param step: global step
        :return:
        """
        # save the session if a path for the saved model is provided
        if saved_model_path and self.save_final_model:
            saver.save(sess, save_path=saved_model_path, global_step=step)

        # print the final losses
        if self.verbose:
            print("Autoencoder Loss: {}".format(autoencoder_loss_final))
            print("Discriminator Loss: {}".format(discriminator_loss_final))
            print("Generator Loss: {}".format(generator_loss_final))

        # set the final performance
        self.final_performance = {"autoencoder_loss_final": autoencoder_loss_final,
                                  "discriminator_loss_final": discriminator_loss_final,
                                  "generator_loss_final": generator_loss_final,
                                  "summed_loss_final": autoencoder_loss_final + discriminator_loss_final
                                                       + generator_loss_final}

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [atoi(c) for c in re.split('(\d+)', text)]

        result_path = self.results_path + self.result_folder_name + '/Tensorboard/'
        filenames = glob.glob(result_path + "*_latent_space_class_distribution.png")
        filenames.sort(key=natural_keys)
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimwrite(result_path + 'latent_space_class_distribution.gif', images, duration=1.0)

        # close the tensorflow session
        sess.close()
        # tf.reset_default_graph()