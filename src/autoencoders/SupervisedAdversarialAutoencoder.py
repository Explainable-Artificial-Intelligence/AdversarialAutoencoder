"""
    Implementation of a supervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""
import json

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin

import util.AdversarialAutoencoderHelperFunctions as aae_helper
from util.Distributions import draw_from_multiple_gaussians, draw_from_single_gaussian, draw_from_swiss_roll


class SupervisedAdversarialAutoencoder(BaseEstimator, TransformerMixin):

    def __init__(self, parameter_dictionary):

        # vars for the swagger server
        self.requested_operations_by_swagger = []
        self.requested_operations_by_swagger_results = None



        self.train_status = "start"
        self.result_folder_name = None
        self.parameter_dictionary = parameter_dictionary
        self.verbose = parameter_dictionary["verbose"]
        self.save_final_model = parameter_dictionary["save_final_model"]        # whether to save the final model
        self.write_tensorboard = parameter_dictionary["write_tensorboard"]      # whether to write the tensorboard file
        # create a summary image of the learning process every n epochs
        self.summary_image_frequency = parameter_dictionary["summary_image_frequency"]

        """
        params for the data 
        """

        # TODO: include in parameter dictionary
        self.n_classes = 10

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
        self.n_neurons_of_hidden_layer_x_discriminator = \
            parameter_dictionary["n_neurons_of_hidden_layer_x_discriminator"]

        # initial bias values of the hidden layers
        self.bias_init_value_of_hidden_layer_x_autoencoder = \
            parameter_dictionary["bias_init_value_of_hidden_layer_x_autoencoder"]
        self.bias_init_value_of_hidden_layer_x_discriminator = \
            parameter_dictionary["bias_init_value_of_hidden_layer_x_discriminator"]

        # activation functions for the different parts of the network
        if type(parameter_dictionary["activation_function_encoder"]) is list:
            self.activation_function_encoder = parameter_dictionary["activation_function_encoder"]
        else:
            self.activation_function_encoder = [parameter_dictionary["activation_function_encoder"]] * \
                                               (len(self.n_neurons_of_hidden_layer_x_autoencoder)+1)

        if type(parameter_dictionary["activation_function_decoder"]) is list:
            self.activation_function_decoder = parameter_dictionary["activation_function_decoder"]
        else:
            self.activation_function_decoder = [parameter_dictionary["activation_function_decoder"]] * \
                                               (len(self.n_neurons_of_hidden_layer_x_autoencoder)+1)

        if type(parameter_dictionary["activation_function_discriminator"]) is list:
            self.activation_function_discriminator = parameter_dictionary["activation_function_discriminator"]
        else:
            self.activation_function_discriminator \
                = [parameter_dictionary["activation_function_discriminator"]] * \
                  (len(self.n_neurons_of_hidden_layer_x_discriminator)+1)

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
        self.global_step_autoencoder = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_discriminator = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_generator = tf.Variable(0, name='global_step', trainable=False)

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
            self.MomentumOptimizer_use_nesterov_autoencoder = \
                parameter_dictionary["MomentumOptimizer_use_nesterov_autoencoder"]

            self.MomentumOptimizer_momentum_discriminator = parameter_dictionary["MomentumOptimizer_momentum_discriminator"]
            self.MomentumOptimizer_use_nesterov_discriminator = \
                parameter_dictionary["MomentumOptimizer_use_nesterov_discriminator"]

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
        loss functions
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
        # holds the labels
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_classes], name='Labels')
        # holds the desired output of the autoencoder
        self.X_target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Target')
        # holds the real distribution p(z) used as positive sample for the discriminator
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim],
                                                name='Real_distribution')
        # holds the input samples for the decoder (only for generating the images; NOT used for training)
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim + self.n_classes],
                                            name='Decoder_input')
        self.decoder_input_multiple = \
            tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim + self.n_classes],
                           name='Decoder_input_multiple')

        """
        Init the network; generator doesn't need to be initiated, since the generator is the encoder of the autoencoder
        """

        # init autoencoder
        with tf.variable_scope(tf.get_variable_scope()):
            # encoder part of the autoencoder and also the generator
            self.encoder_output = self.encoder(self.X, bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # Concat class label and the encoder output
            decoder_input = tf.concat([self.y, self.encoder_output], 1)
            # decoder part of the autoencoder
            self.decoder_output = self.decoder(decoder_input,
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
            # used for "manually" passing single values through the decoder
            self.decoder_output_real_dist = self.decoder(self.decoder_input, reuse=True,
                                                         bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # used for "manually" passing multiple values through the decoder
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
            get_optimizer(self, optimizer_autoencoder, "autoencoder", global_step=self.global_step_autoencoder,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.autoencoder_trainer = self.autoencoder_optimizer.minimize(self.autoencoder_loss,
                                                                       global_step=self.global_step_autoencoder)
        self.discriminator_optimizer = aae_helper. \
            get_optimizer(self, optimizer_discriminator, "discriminator", global_step=self.global_step_discriminator,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator)
        self.discriminator_trainer = self.discriminator_optimizer.minimize(self.discriminator_loss,
                                                                       var_list=discriminator_vars,
                                                                       global_step=self.global_step_discriminator)
        self.generator_optimizer = aae_helper. \
            get_optimizer(self, optimizer_generator, "generator", global_step=self.global_step_generator,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_generator)
        self.generator_trainer = self.generator_optimizer.minimize(self.generator_loss, var_list=encoder_vars,
                                                                   global_step=self.global_step_generator)

        """
        Create the tensorboard summary and the tf.saver and tf.session vars
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=self.decoder_output, encoder_output=self.encoder_output,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_loss=self.discriminator_loss,
                                            generator_loss=self.generator_loss,
                                            real_distribution=self.real_distribution,
                                            decoder_output_multiple=self.decoder_output_multiple)

        # for saving the model
        self.saver = tf.train.Saver()

        self.session = tf.Session()

        """
        Variable for the "manual" summary
        """
        self.final_performance = None
        self.performance_over_time = {"autoencoder_losses": [], "discriminator_losses": [], "generator_losses": [],
                                      "list_of_epochs": []}
        self.learning_rates = {"autoencoder_lr": [], "discriminator_lr": [], "generator_lr": [], "list_of_epochs": []}

        # variables for the minibatch summary image
        self.minibatch_summary_vars = {"real_dist": None, "latent_representation": None, "discriminator_neg": None,
                                       "discriminator_pos": None, "batch_x": None, "decoder_output": None,
                                       "epoch": None, "b": None, "batch_labels": None}

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    def get_requested_operations_by_swagger_results(self):
        return self.requested_operations_by_swagger_results

    def set_requested_operations_by_swagger_results(self, requested_operations_by_swagger_results):
        self.requested_operations_by_swagger_results = requested_operations_by_swagger_results

    def get_requested_operations_by_swagger(self):
        return self.requested_operations_by_swagger

    def add_to_requested_operations_by_swagger(self, requested_operation):
        self.requested_operations_by_swagger.append(requested_operation)

    def get_minibatch_summary_vars(self):
        return self.minibatch_summary_vars

    def get_performance_over_time(self):
        return self.performance_over_time

    def get_learning_rates(self):
        return self.learning_rates

    def get_train_status(self):
        return self.train_status

    def set_train_status(self, status_to_set):
        self.train_status = status_to_set

    def get_final_performance(self):
        return self.final_performance

    def get_performance(self):
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
                latent_variable = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[0], aae_helper.\
                    create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                       bias_init_value=bias_init_values[0]))
                return latent_variable
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[0],
                    aae_helper.
                        create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                latent_variable = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[-1], aae_helper.\
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       self.z_dim, 'encoder_output', bias_init_value=bias_init_values[1]))
                return latent_variable
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[0],
                    aae_helper.
                        create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n_hidden_layers):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[i],
                        aae_helper.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                               self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                               'encoder_dense_layer_' + str(i + 1),
                                               bias_init_value=bias_init_values[i]))
                latent_variable = aae_helper.use_activation_function_for_layer(self.activation_function_encoder[-1], aae_helper.\
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[-1], self.z_dim,
                                       'encoder_output', bias_init_value=bias_init_values[-1]))
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
                decoder_output = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[0],
                    aae_helper.
                        create_dense_layer(X, self.z_dim + self.n_classes, self.input_dim, 'decoder_output',
                                           bias_init_value=bias_init_values[0]))
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[0],
                    aae_helper.
                        create_dense_layer(X, self.z_dim + self.n_classes,
                                           self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                decoder_output = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[-1],
                    aae_helper.
                        create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                           self.input_dim, 'decoder_output', bias_init_value=bias_init_values[1]))
                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[0],
                    aae_helper.
                        create_dense_layer(X, self.z_dim + self.n_classes,
                                           self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                           'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[i],
                        aae_helper.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                               self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                               'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                               bias_init_value=bias_init_values[i]))
                decoder_output = aae_helper.use_activation_function_for_layer(self.activation_function_decoder[-1],
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
                discriminator_output = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[0], aae_helper.\
                    create_dense_layer(X, self.z_dim, 1, 'discriminator_output', bias_init_value=bias_init_values[0]))
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[0],
                    aae_helper.
                        create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                discriminator_output = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[-1], aae_helper.\
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                       'discriminator_output', bias_init_value=bias_init_values[1]))
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[0],
                    aae_helper.
                        create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                           'discriminator_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[i],
                        aae_helper.
                            create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                               self.n_neurons_of_hidden_layer_x_discriminator[i],
                                               'discriminator_dense_layer_' + str(i + 1),
                                               bias_init_value=bias_init_values[i]))
                discriminator_output = aae_helper.use_activation_function_for_layer(self.activation_function_discriminator[-1], aae_helper.\
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                       'discriminator_output', bias_init_value=bias_init_values[-1]))
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

        # Reshape images accordingly to the color scale to display them
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
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param left_cell: left cell of the grid spec with two adjacent horizontal cells holding the image grid
        and the class distribution on the latent space; if left_cell is None, then only the image grid is supposed
        to be plotted and not the "combinated" image (image grid + class distr. on latent space).
        :return: None, displays a matplotlib window with all the merged images.
        """
        nx, ny = self.n_classes, self.n_classes
        random_inputs = np.random.randn(self.n_classes, self.z_dim) * 5.

        class_labels = np.identity(self.n_classes)

        # create the image grid
        if left_cell:
            gs = gridspec.GridSpecFromSubplotSpec(nx, ny, left_cell)
        else:
            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        i = 0
        for class_label_one_hot in class_labels:
            for r in random_inputs:
                r = np.reshape(r, (1, self.z_dim))
                class_label_one_hot = np.reshape(class_label_one_hot, (1, self.n_classes))
                dec_input = np.concatenate((class_label_one_hot, r), 1)
                x = sess.run(op, feed_dict={self.decoder_input: dec_input})
                ax = plt.subplot(gs[i])
                i += 1

                # reshape the images according to the color scale
                img = aae_helper.reshape_image_array(self, x, is_array_of_arrays=True)

                # show the image
                if self.color_scale == "gray_scale":
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('auto')

                # create the label for the y axis
                if ax.is_first_col():
                    class_label = int(i / self.n_classes)
                    ax.set_ylabel(class_label, fontsize=9)

        if not left_cell:
            plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')
        # plt.show()

    def generate_image_from_single_point_and_class_label(self, sess, params):
        """
        generates a image based on the point on the latent space and the class label
        :param sess: tensorflow session
        :param params:
        :return:
        """

        # get the point and the class label
        single_point = params[0]
        class_label_one_hot = params[1]

        # reshape the point and the class label
        single_point = np.reshape(single_point, (1, self.z_dim))
        class_label_one_hot = np.reshape(class_label_one_hot, (1, self.n_classes))

        # create the decoder input
        dec_input = np.concatenate((class_label_one_hot, single_point), 1)

        generated_image = sess.run(self.decoder_output_real_dist, feed_dict={self.decoder_input: dec_input})

        generated_image = np.array(generated_image).reshape(self.input_dim)

        # reshape the image array and display it
        img = aae_helper.reshape_image_array(self, generated_image, is_array_of_arrays=True)

        # show the image
        if self.color_scale == "gray_scale":
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + "testing" + '.png')


        return img

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        # we need a new session, since training has been completed and the old session is closed
        if not is_train_mode_active:
            self.session = tf.Session()

        saved_model_path = None

        latent_representations_current_epoch = []
        labels_current_epoch = []

        # Get the data
        data = aae_helper.get_input_data(self.selected_dataset)

        autoencoder_loss_final, discriminator_loss_final, generator_loss_final = 0, 0, 0

        step = 0
        with self.session as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:
                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = aae_helper.form_results(self)
                if self.write_tensorboard:
                    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # write the parameter dictionary to some file
                json_dictionary = json.dumps(self.parameter_dictionary)
                with open(log_path + '/params.txt', 'a') as file:
                    file.write(json_dictionary)

                # we want n_epochs iterations
                for epoch in range(self.n_epochs):

                    if self.train_status == "stop":
                        # end the training
                        break

                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(data.train.num_examples / self.batch_size)

                    if self.verbose:
                        print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        self.process_requested_swagger_operations(sess, "testing")

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        z_real_dist = draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(self.batch_size, self.z_dim))

                        # get the batch from the training data
                        batch_x, batch_labels = data.train.next_batch(self.batch_size)
                        # print(batch_x)

                        """
                        Reconstruction phase: the autoencoder updates the encoder and the decoder to minimize the
                        reconstruction error of the inputs
                        """
                        # train the autoencoder by minimizing the reconstruction error between X and X_target
                        sess.run(self.autoencoder_trainer, feed_dict={self.X: batch_x, self.X_target: batch_x,
                                                                        self.y: batch_labels})

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

                        # every x epochs: write a summary for every 50th minibatch
                        if epoch % self.summary_image_frequency == 0 and b % 50 == 0:

                            # prepare the decoder inputs
                            n_images_per_class = int(math.ceil(self.batch_size / self.n_classes))
                            class_labels_one_hot = np.identity(self.n_classes)
                            dec_inputs = []
                            for k in range(n_images_per_class):
                                for class_label in class_labels_one_hot:
                                    random_inputs = np.random.randn(1, self.z_dim) * 5.
                                    random_inputs = np.reshape(random_inputs, (1, self.z_dim))
                                    class_label = np.reshape(class_label, (1, self.n_classes))
                                    dec_inputs.append(np.concatenate((random_inputs, class_label), 1))
                            dec_inputs = dec_inputs[:self.batch_size]
                            dec_inputs = np.array(dec_inputs).reshape(self.batch_size, self.z_dim+self.n_classes)

                            # get the network output for the summary images
                            autoencoder_loss, discriminator_loss, generator_loss, summary, real_dist, \
                            latent_representation, discriminator_neg, discriminator_pos, decoder_output = sess.run(
                                [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                 self.tensorboard_summary, self.real_distribution, self.encoder_output,
                                 self.discriminator_neg_samples, self.discriminator_pos_samples,
                                 self.decoder_output],
                                feed_dict={self.X: batch_x, self.X_target: batch_x,
                                           self.real_distribution: z_real_dist, self.y: batch_labels,
                                           self.decoder_input_multiple: dec_inputs})
                            if self.write_tensorboard:
                                writer.add_summary(summary, global_step=step)

                            # prepare the decoder input for a single input image
                            r = np.reshape(latent_representation[0, :], (1, self.z_dim))
                            class_label_one_hot = np.reshape(batch_labels[0, :], (1, self.n_classes))
                            dec_input = np.concatenate((class_label_one_hot, r), 1)

                            # reconstruct the image
                            reconstructed_image = sess.run(self.decoder_output_real_dist,
                                                           feed_dict={self.decoder_input: dec_input})

                            # update the dictionary holding the losses
                            self.performance_over_time["autoencoder_losses"].append(autoencoder_loss)
                            self.performance_over_time["discriminator_losses"].append(discriminator_loss)
                            self.performance_over_time["generator_losses"].append(generator_loss)
                            self.performance_over_time["list_of_epochs"].append(epoch + (b / n_batches))

                            # update the dictionary holding the learning rates
                            self.learning_rates["autoencoder_lr"].append(
                                aae_helper.get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess))
                            self.learning_rates["discriminator_lr"].append(
                                aae_helper.get_learning_rate_for_optimizer(self.discriminator_optimizer, sess))
                            self.learning_rates["generator_lr"].append(
                                aae_helper.get_learning_rate_for_optimizer(self.generator_optimizer, sess))
                            self.learning_rates["list_of_epochs"].append(epoch + (b / n_batches))

                            # update the lists holding the latent representation + labels for the current minibatch
                            latent_representations_current_epoch.extend(latent_representation)
                            labels_current_epoch.extend(batch_labels)

                            # updates vars for the swagger server
                            self.minibatch_summary_vars["real_dist"] = real_dist
                            self.minibatch_summary_vars["latent_representation"] = latent_representation
                            self.minibatch_summary_vars["discriminator_neg"] = discriminator_neg
                            self.minibatch_summary_vars["discriminator_pos"] = discriminator_pos
                            self.minibatch_summary_vars["batch_x"] =  batch_x
                            self.minibatch_summary_vars["decoder_output"] = decoder_output
                            self.minibatch_summary_vars["epoch"] = epoch
                            self.minibatch_summary_vars["b"] = b
                            self.minibatch_summary_vars["batch_labels"] = batch_labels

                            # create the summary image for the current minibatch
                            aae_helper.create_minibatch_summary_image(self, real_dist, latent_representation,
                                                                      discriminator_neg, discriminator_pos, batch_x,
                                                                      reconstructed_image, epoch, b, batch_labels)

                            # set the latest loss as final loss
                            autoencoder_loss_final = autoencoder_loss
                            discriminator_loss_final = discriminator_loss
                            generator_loss_final = generator_loss

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(epoch, b))
                                print("summed losses:", autoencoder_loss + discriminator_loss + generator_loss)
                                print("Autoencoder Loss: {}".format(autoencoder_loss))
                                print("Discriminator Loss: {}".format(discriminator_loss))
                                print("Generator Loss: {}".format(generator_loss))
                                print('Learning rate autoencoder: {}'.format(
                                    aae_helper.get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess)))
                                print('Learning rate discriminator: {}'.format(
                                    aae_helper.get_learning_rate_for_optimizer(self.discriminator_optimizer, sess)))
                                print('Learning rate generator: {}'.format(
                                    aae_helper.get_learning_rate_for_optimizer(self.generator_optimizer, sess)))

                            with open(log_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                                log.write("Autoencoder Loss: {}\n".format(autoencoder_loss))
                                log.write("Discriminator Loss: {}\n".format(discriminator_loss))
                                log.write("Generator Loss: {}\n".format(generator_loss))
                        step += 1

                    # every x epochs ..
                    if epoch % self.summary_image_frequency == 0:

                        # increase figure size
                        plt.rcParams["figure.figsize"] = (6.4*2, 4.8)
                        outer_grid = gridspec.GridSpec(1, 2)
                        left_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid

                        # generate the image grid for the latent space
                        self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=epoch,
                                                 left_cell=left_cell)
                        result_path = self.results_path + self.result_folder_name + '/Tensorboard/'

                        # draw the class distribution on the latent space
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
                self.saver.restore(sess, save_path=tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1]
                                                                              + '/Saved_models/'))

                self.process_requested_swagger_operations(sess, "testing")

                self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch="last")

            # end the training
            self.end_training(autoencoder_loss_final, discriminator_loss_final, generator_loss_final,
                              saved_model_path, self.saver, sess, step)

    def process_requested_swagger_operations(self, sess, image_title):
        """
        processes the operations requested by swagger
        :param sess: tensorflow session
        :param image_title: title of the image
        :return:
        """

        # check if any operations to run are requested by swagger
        if len(self.requested_operations_by_swagger) > 0:
            # requested_operations_by_swagger is a list of dicts: {"function_name_1": [params],
            # "function_name_2": [params]}
            for requested_operation in self.requested_operations_by_swagger:
                for key, value in requested_operation.items():

                    # get the function name and the function parameters
                    function_name = key
                    function_params = value

                    print(function_name)
                    print(function_params)

                    # call the respective function with the respective parameters
                    if function_name == "generate_image_grid":
                        self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=image_title,
                                                 left_cell=None)
                    elif function_name == "generate_image_from_single_point_and_single_label":
                        result = self.generate_image_from_single_point_and_class_label(sess, function_params)
                        self.set_requested_operations_by_swagger_results(result)

                    plt.close('all')

            # reset the list
            self.requested_operations_by_swagger = []

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
            print("#############    FINISHED TRAINING   #############")

        # set the final performance
        self.final_performance = {"autoencoder_loss_final": autoencoder_loss_final,
                                  "discriminator_loss_final": discriminator_loss_final,
                                  "generator_loss_final": generator_loss_final,
                                  "summed_loss_final": autoencoder_loss_final + discriminator_loss_final
                                                       + generator_loss_final}

        # create the gif for the learning progress
        aae_helper.create_gif(self)

        # training has stopped
        self.train_status = "stop"

        # close the tensorflow session
        sess.close()
        # tf.reset_default_graph()