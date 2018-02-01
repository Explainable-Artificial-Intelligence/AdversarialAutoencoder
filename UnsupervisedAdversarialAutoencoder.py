"""
    Implementation of an Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""
import json

import tensorflow as tf
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin
import DataLoading


class AdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        # TODO: remove this; only for testing
        # parameter_dictionary["color_scale"] = "gray_scale"

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
        self.bias_init_value_of_hidden_layer_x_autoencoder = parameter_dictionary[
            "bias_init_value_of_hidden_layer_x_autoencoder"]
        self.bias_init_value_of_hidden_layer_x_discriminator = parameter_dictionary[
            "bias_init_value_of_hidden_layer_x_discriminator"]

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
            self.get_optimizer_autoencoder(optimizer_autoencoder).minimize(self.autoencoder_loss)
        self.discriminator_optimizer = \
            self.get_optimizer_autoencoder(optimizer_discriminator).minimize(self.discriminator_loss,
                                                                             var_list=discriminator_vars)
        self.generator_optimizer = \
            self.get_optimizer_autoencoder(optimizer_generator).minimize(self.generator_loss,
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

    def get_optimizer_autoencoder(self, optimizer_name):

        if optimizer_name == "GradientDescentOptimizer":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_autoencoder)
        elif optimizer_name == "AdadeltaOptimizer":
            return tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate_autoencoder, rho=self.AdadeltaOptimizer_rho_autoencoder,
                epsilon=self.AdadeltaOptimizer_epsilon_autoencoder)
        elif optimizer_name == "AdagradOptimizer":
            return tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate_autoencoder,
                initial_accumulator_value=self.AdagradOptimizer_initial_accumulator_value_autoencoder),
        elif optimizer_name == "MomentumOptimizer":
            return tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate_autoencoder, momentum=self.MomentumOptimizer_momentum_autoencoder,
                use_nesterov=self.MomentumOptimizer_use_nesterov_autoencoder)
        elif optimizer_name == "AdamOptimizer":
            return tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_autoencoder, beta1=self.AdamOptimizer_beta1_autoencoder,
                beta2=self.AdamOptimizer_beta2_autoencoder, epsilon=self.AdamOptimizer_epsilon_autoencoder)
        elif optimizer_name == "FtrlOptimizer":
            return tf.train.FtrlOptimizer(
                learning_rate=self.learning_rate_autoencoder,
                learning_rate_power=self.FtrlOptimizer_learning_rate_power_autoencoder,
                initial_accumulator_value=self.FtrlOptimizer_initial_accumulator_value_autoencoder,
                l1_regularization_strength=self.FtrlOptimizer_l1_regularization_strength_autoencoder,
                l2_regularization_strength=self.FtrlOptimizer_l2_regularization_strength_autoencoder,
                l2_shrinkage_regularization_strength=self.FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder
            )
        elif optimizer_name == "ProximalGradientDescentOptimizer":
            return tf.train.ProximalGradientDescentOptimizer(
                learning_rate=self.learning_rate_autoencoder,
                l1_regularization_strength=self.ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder,
                l2_regularization_strength=self.ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder
            )
        elif optimizer_name == "ProximalAdagradOptimizer":
            return tf.train.ProximalAdagradOptimizer(
                learning_rate=self.learning_rate_autoencoder,
                initial_accumulator_value=self.ProximalAdagradOptimizer_initial_accumulator_value_autoencoder,
                l1_regularization_strength=self.ProximalAdagradOptimizer_l1_regularization_strength_autoencoder,
                l2_regularization_strength=self.ProximalAdagradOptimizer_l2_regularization_strength_autoencoder
            )
        elif optimizer_name == "RMSPropOptimizer":
            return tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate_autoencoder, decay=self.RMSPropOptimizer_decay_autoencoder,
                momentum=self.RMSPropOptimizer_momentum_autoencoder, epsilon=self.RMSPropOptimizer_epsilon_autoencoder,
                centered=self.RMSPropOptimizer_centered_autoencoder)

    def get_optimizer_discriminator(self, optimizer_name):

        if optimizer_name == "GradientDescentOptimizer":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_discriminator)
        elif optimizer_name == "AdadeltaOptimizer":
            return tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate_discriminator, rho=self.AdadeltaOptimizer_rho_discriminator,
                epsilon=self.AdadeltaOptimizer_epsilon_discriminator)
        elif optimizer_name == "AdagradOptimizer":
            return tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate_discriminator,
                initial_accumulator_value=self.AdagradOptimizer_initial_accumulator_value_discriminator),
        elif optimizer_name == "MomentumOptimizer":
            return tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate_discriminator, momentum=self.MomentumOptimizer_momentum_discriminator,
                use_nesterov=self.MomentumOptimizer_use_nesterov_discriminator)
        elif optimizer_name == "AdamOptimizer":
            return tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_discriminator, beta1=self.AdamOptimizer_beta1_discriminator,
                beta2=self.AdamOptimizer_beta2_discriminator, epsilon=self.AdamOptimizer_epsilon_discriminator)
        elif optimizer_name == "FtrlOptimizer":
            return tf.train.FtrlOptimizer(
                learning_rate=self.learning_rate_discriminator,
                learning_rate_power=self.FtrlOptimizer_learning_rate_power_discriminator,
                initial_accumulator_value=self.FtrlOptimizer_initial_accumulator_value_discriminator,
                l1_regularization_strength=self.FtrlOptimizer_l1_regularization_strength_discriminator,
                l2_regularization_strength=self.FtrlOptimizer_l2_regularization_strength_discriminator,
                l2_shrinkage_regularization_strength=self.FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator
            )
        elif optimizer_name == "ProximalGradientDescentOptimizer":
            return tf.train.ProximalGradientDescentOptimizer(
                learning_rate=self.learning_rate_discriminator,
                l1_regularization_strength=self.ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator,
                l2_regularization_strength=self.ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator
            )
        elif optimizer_name == "ProximalAdagradOptimizer":
            return tf.train.ProximalAdagradOptimizer(
                learning_rate=self.learning_rate_discriminator,
                initial_accumulator_value=self.ProximalAdagradOptimizer_initial_accumulator_value_discriminator,
                l1_regularization_strength=self.ProximalAdagradOptimizer_l1_regularization_strength_discriminator,
                l2_regularization_strength=self.ProximalAdagradOptimizer_l2_regularization_strength_discriminator
            )
        elif optimizer_name == "RMSPropOptimizer":
            return tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate_discriminator, decay=self.RMSPropOptimizer_decay_discriminator,
                momentum=self.RMSPropOptimizer_momentum_discriminator,
                epsilon=self.RMSPropOptimizer_epsilon_discriminator,
                centered=self.RMSPropOptimizer_centered_discriminator)

    def get_optimizer_generator(self, optimizer_name):

        if optimizer_name == "GradientDescentOptimizer":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_generator)
        elif optimizer_name == "AdadeltaOptimizer":
            return tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate_generator, rho=self.AdadeltaOptimizer_rho_generator,
                epsilon=self.AdadeltaOptimizer_epsilon_generator)
        elif optimizer_name == "AdagradOptimizer":
            return tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate_generator,
                initial_accumulator_value=self.AdagradOptimizer_initial_accumulator_value_generator),
        elif optimizer_name == "MomentumOptimizer":
            return tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate_generator, momentum=self.MomentumOptimizer_momentum_generator,
                use_nesterov=self.MomentumOptimizer_use_nesterov_generator)
        elif optimizer_name == "AdamOptimizer":
            return tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_generator, beta1=self.AdamOptimizer_beta1_generator,
                beta2=self.AdamOptimizer_beta2_generator, epsilon=self.AdamOptimizer_epsilon_generator)
        elif optimizer_name == "FtrlOptimizer":
            return tf.train.FtrlOptimizer(
                learning_rate=self.learning_rate_generator,
                learning_rate_power=self.FtrlOptimizer_learning_rate_power_generator,
                initial_accumulator_value=self.FtrlOptimizer_initial_accumulator_value_generator,
                l1_regularization_strength=self.FtrlOptimizer_l1_regularization_strength_generator,
                l2_regularization_strength=self.FtrlOptimizer_l2_regularization_strength_generator,
                l2_shrinkage_regularization_strength=self.FtrlOptimizer_l2_shrinkage_regularization_strength_generator
            )
        elif optimizer_name == "ProximalGradientDescentOptimizer":
            return tf.train.ProximalGradientDescentOptimizer(
                learning_rate=self.learning_rate_generator,
                l1_regularization_strength=self.ProximalGradientDescentOptimizer_l1_regularization_strength_generator,
                l2_regularization_strength=self.ProximalGradientDescentOptimizer_l2_regularization_strength_generator
            )
        elif optimizer_name == "ProximalAdagradOptimizer":
            return tf.train.ProximalAdagradOptimizer(
                learning_rate=self.learning_rate_generator,
                initial_accumulator_value=self.ProximalAdagradOptimizer_initial_accumulator_value_generator,
                l1_regularization_strength=self.ProximalAdagradOptimizer_l1_regularization_strength_generator,
                l2_regularization_strength=self.ProximalAdagradOptimizer_l2_regularization_strength_generator
            )
        elif optimizer_name == "RMSPropOptimizer":
            return tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate_generator, decay=self.RMSPropOptimizer_decay_generator,
                momentum=self.RMSPropOptimizer_momentum_generator, epsilon=self.RMSPropOptimizer_epsilon_generator,
                centered=self.RMSPropOptimizer_centered_generator)

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

        if loss_function == "hinge_loss":
            return tf.losses.hinge_loss(labels=labels, logits=logits)
        elif loss_function == "mean_squared_error":
            return tf.losses.mean_squared_error(labels=labels, predictions=logits)
        elif loss_function == "sigmoid_cross_entropy":
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        elif loss_function == "softmax_cross_entropy":
            return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits),

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

    @staticmethod
    def reshape_to_rgb_image(image_array, input_dim_x, input_dim_y):
        """
        reshapes the given image array to an rgb image for the tensorboard visualization
        :param image_array: array of images to be reshaped
        :param input_dim_x: dim x of the images
        :param input_dim_y: dim y of the images
        :return:
        """
        n_colored_pixels_per_channel = input_dim_x * input_dim_y
        red_pixels = tf.reshape(image_array[:, :n_colored_pixels_per_channel],
                                [-1, input_dim_x, input_dim_y, 1])
        green_pixels = tf.reshape(image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2],
                                  [-1, input_dim_x, input_dim_y, 1])
        blue_pixels = tf.reshape(image_array[:, n_colored_pixels_per_channel * 2:],
                                 [-1, input_dim_x, input_dim_y, 1])
        rgb_image = tf.concat([red_pixels, green_pixels, blue_pixels], 3)
        return rgb_image

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
            input_images = self.reshape_to_rgb_image(self.X, self.input_dim_x, self.input_dim_y)
            generated_images = self.reshape_to_rgb_image(decoder_output, self.input_dim_x, self.input_dim_y)
            generated_images_z_dist = self.reshape_to_rgb_image(decoder_output_multiple, self.input_dim_x,
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

    def form_results(self):
        """
        Forms folders for each run to store the tensorboard files, saved models and the log files.
        :return: three strings pointing to tensorboard, saved models and log paths respectively.
        """
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"). \
            replace(" ", ":").replace(":", "_")

        folder_name = "/{0}_{1}". \
            format(date, self.selected_dataset)
        self.result_folder_name = folder_name
        tensorboard_path = self.results_path + folder_name + '/Tensorboard'
        saved_model_path = self.results_path + folder_name + '/Saved_models/'
        log_path = self.results_path + folder_name + '/log'

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        if not os.path.exists(self.results_path + folder_name):
            os.mkdir(self.results_path + folder_name)
            os.mkdir(tensorboard_path)
            os.mkdir(saved_model_path)
            os.mkdir(log_path)
        return tensorboard_path, saved_model_path, log_path

    def generate_image_grid(self, sess, op, epoch):
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

        print(self.color_scale)

        # TODO: adjust function that it also works with RGB data

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

            if self.color_scale == "gray_scale":
                img = np.array(x.tolist()).reshape(self.input_dim_x, self.input_dim_y)
                ax.imshow(img, cmap='gray')
            else:
                image_array = np.array(x.tolist())
                n_colored_pixels_per_channel = self.input_dim_x * self.input_dim_y

                red_pixels = image_array[:, :n_colored_pixels_per_channel].reshape(32, 32, 1)
                green_pixels = \
                    image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel*2].reshape(32, 32, 1)
                blue_pixels = image_array[:, n_colored_pixels_per_channel*2:].reshape(32, 32, 1)
                img = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)
                ax.imshow(img)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            # label y
            if ax.is_first_col():
                ax.set_ylabel(x_points[int(i / ny)], fontsize=9)

            # label x
            if ax.is_last_row():
                ax.set_xlabel(y_points[int(i % ny)], fontsize=9)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')
        # plt.show()

    def generate_image_grid_z_dim(self, sess, op, epoch):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :return: None, displays a matplotlib window with all the merged images.
        """

        # TODO: draw with function, not hard coded
        random_points = [np.arange(-10, 10, 1.5).astype(np.float32) for i in range(self.z_dim)]

        # creates evenly spaced values within [-10, 10] with a spacing of 1.5
        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        # TODO: adjust function that it also works with RGB data

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
            img = np.array(x.tolist()).reshape(self.input_dim_x, self.input_dim_y)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            # label y
            if ax.is_first_col():
                ax.set_ylabel(x_points[int(i / ny)], fontsize=9)

            # label x
            if ax.is_last_row():
                ax.set_xlabel(y_points[int(i % ny)], fontsize=9)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '_.png')
        # plt.show()

    @staticmethod
    def get_input_data(selected_dataset):
        """
        returns the input data set based on self.selected_dataset
        :return: object holding the train data, the test data and the validation data
        """

        # Modified National Institute of Standards and Technology
        if selected_dataset == "MNIST":
            return DataLoading.read_mnist_data_from_ubyte('./data', one_hot=True)
        # Street View House Numbers
        elif selected_dataset == "SVHN":
            return DataLoading.read_svhn_from_mat('./data', one_hot=True, one_channel=False)
            # TODO: remove this; only for testing
            # return DataLoading.read_svhn_from_mat('./data', one_hot=True, one_channel=True)
        elif selected_dataset == "cifar10":
            return DataLoading.read_cifar10('./data', one_hot=True)
        elif selected_dataset == "custom":
            # TODO:
            print("not yet implemented")
            return

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        log_path = None

        # Get the data
        data = self.get_input_data(self.selected_dataset)

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
                tensorboard_path, saved_model_path, log_path = self.form_results()
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

                        # every 50 steps: write a summary
                        if b % 50 == 0:

                            # TODO: depending on batch_size

                            # creates evenly spaced values within [-10, 10] with a spacing of 1.5
                            x_points = np.arange(-9, 10, 2.0).astype(np.float32)
                            y_points = np.arange(-9, 10, 2.0).astype(np.float32)
                            nx, ny = len(x_points), len(y_points)
                            points = []

                            for j in range(nx):
                                for k in range(ny):
                                    # create a data point from the x_points and y_points array as input for the decoder
                                    # z = np.concatenate(([x_points[j]], [y_points[k]]))
                                    z = np.array([x_points[j], y_points[k]])
                                    # z = np.reshape(z, (1, 2))
                                    points.append(z)

                                    # create a data point from the x_points and y_points array as input for the decoder
                                    # z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
                                    # z = np.reshape(z, (1, 2))

                            points = np.array(points)

                            # random_points = [np.arange(-9, 10, 2.0).astype(np.float32) for i in range(self.z_dim)]
                            # TODO: shape: batch_size, z_dim

                            # TODO: test if it still works when , test and self.decoder_output_multiple get removed
                            a_loss, d_loss, g_loss, summary, test = sess.run(
                                [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                 self.tensorboard_summary, self.decoder_output_multiple],
                                feed_dict={self.X: batch_x, self.X_target: batch_x,
                                           self.real_distribution: z_real_dist,
                                           self.decoder_input_multiple: points})
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

                    if i % 5 == 0:
                        self.generate_image_grid(sess, op=self.decoder_output, epoch=i)
                    # self.generate_image_grid_z_dim(sess, op=self.decoder_output_multiple, epoch=i)

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
