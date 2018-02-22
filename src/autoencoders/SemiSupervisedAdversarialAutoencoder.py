"""
    Implementation of a SemiSupervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
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

import src.util.AdversarialAutoencoderHelperFunctions as aae_helper
from src.util.Distributions import draw_from_multiple_gaussians, draw_from_single_gaussian, draw_from_swiss_roll


class SemiSupervisedAdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        self.train_status = "start"
        self.result_folder_name = None
        self.parameter_dictionary = parameter_dictionary
        self.verbose = parameter_dictionary["verbose"]

        """
        params for the data 
        """

        # TODO: include in parameter dictionary
        self.n_classes = 10
        self.n_labeled = 1000

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
        # holds the labeled input data
        self.X_labeled = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Labeled_Input')
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
        # holds the categorical distribution
        self.categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_classes],
                                                      name='Categorical_distribution')

        """
        Init the network; generator doesn't need to be initiated, since the generator is the encoder of the autoencoder
        """

        # init autoencoder
        with tf.variable_scope(tf.get_variable_scope()):
            # encoder part of the autoencoder and also the generator
            self.latent_variable_z, self.class_label_y = \
                self.encoder(self.X, bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)
            # Concat class label and the encoder output
            decoder_input = tf.concat([self.class_label_y, self.latent_variable_z], 1)
            # decoder part of the autoencoder; takes z and y as input
            decoder_output = self.decoder(decoder_input,
                                          bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        # init discriminator for the samples drawn from some gaussian distribution (holds style information)
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive gaussian samples drawn from N(z|0,I)
            self.discriminator_gaussian_pos_samples = \
                self.discriminator_gaussian(self.real_distribution,
                                            bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)
            # discriminator for the negative gaussian samples q(z) (generated by the generator)
            self.discriminator_gaussian_neg_samples = \
                self.discriminator_gaussian(self.latent_variable_z, reuse=True,
                                            bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)

        # init discriminator for the samples drawn from the categorical distribution (holds class label information)
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive categorical samples drawn from Cat(y)
            discriminator_categorical_pos_samples = \
                self.discriminator_categorical(self.categorial_distribution,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)
            # discriminator for the negative categorical samples (= predicted labels y) (generated by the
            # generator)
            discriminator_categorical_neg_samples = \
                self.discriminator_categorical(self.class_label_y, reuse=True,
                                               bias_init_values=self.bias_init_value_of_hidden_layer_x_discriminator)

        # variable for predicting the class labels from the labeled data (for performance evaluation)
        with tf.variable_scope(tf.get_variable_scope()):
            # predict the labels by passing the data through the encoder part
            _, predicted_labels = \
                self.encoder(self.X_labeled, bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder,
                             reuse=True, is_supervised=True)

        # variable for "manually" passing values through the decoder (currently not in use -> later when clicking on
        # distribution -> show respective image)
        with tf.variable_scope(tf.get_variable_scope()):
            self.decoder_output_real_dist = self.decoder(self.decoder_input, reuse=True,
                                                         bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

            self.decoder_output_multiple = \
                self.decoder(self.decoder_input_multiple, reuse=True,
                             bias_init_values=self.bias_init_value_of_hidden_layer_x_autoencoder)

        # Classification accuracy of encoder
        # compare the predicted labels with the actual labels
        correct_pred = tf.equal(tf.argmax(predicted_labels, 1), tf.argmax(self.y, 1))
        # calculate the accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        """
        Init the loss functions
        """

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - decoder_output))

        # Gaussian Discriminator Loss
        discriminator_gaussian_loss_pos_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.ones_like(self.discriminator_gaussian_pos_samples),
                                         logits=self.discriminator_gaussian_pos_samples))
        discriminator_gaussian_loss_neg_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.zeros_like(self.discriminator_gaussian_neg_samples),
                                         logits=self.discriminator_gaussian_neg_samples))
        self.discriminator_gaussian_loss = discriminator_gaussian_loss_neg_samples + \
                                           discriminator_gaussian_loss_pos_samples

        # Categorical Discrimminator Loss
        discriminator_categorical_loss_pos_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.ones_like(discriminator_categorical_pos_samples),
                                         logits=discriminator_categorical_pos_samples))
        discriminator_categorical_loss_neg_samples = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_discriminator,
                                         labels=tf.zeros_like(discriminator_categorical_neg_samples),
                                         logits=discriminator_categorical_neg_samples))
        self.discriminator_categorical_loss = discriminator_categorical_loss_pos_samples + \
                                              discriminator_categorical_loss_neg_samples

        # Generator loss
        generator_gaussian_loss = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_generator,
                                         labels=tf.ones_like(self.discriminator_gaussian_neg_samples),
                                         logits=self.discriminator_gaussian_neg_samples))
        generator_categorical_loss = tf.reduce_mean(
            aae_helper.get_loss_function(loss_function=self.loss_function_generator,
                                         labels=tf.ones_like(discriminator_categorical_neg_samples),
                                         logits=discriminator_categorical_neg_samples))
        self.generator_loss = generator_gaussian_loss + generator_categorical_loss

        # Supervised Encoder Loss
        self.supervised_encoder_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=predicted_labels))

        """
        Init the optimizers
        """

        optimizer_autoencoder = parameter_dictionary["optimizer_autoencoder"]
        optimizer_discriminator = parameter_dictionary["optimizer_discriminator"]
        optimizer_generator = parameter_dictionary["optimizer_generator"]

        # get the discriminator and encoder variables
        all_variables = tf.trainable_variables()
        discriminator_gaussian_vars = [var for var in all_variables if 'discriminator_gaussian' in var.name]
        discriminator_categorical_vars = [var for var in all_variables if 'discriminator_categorical' in var.name]
        encoder_vars = [var for var in all_variables if 'encoder_' in var.name]

        # Optimizers
        # TODO: include optimizer in param dict (add discriminator_gaussian, discriminator_categorical,
        # supervised_encoder_optimizer, remove discriminator)
        self.autoencoder_optimizer = aae_helper.\
            get_optimizer(self, optimizer_autoencoder, "autoencoder", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.autoencoder_trainer = self.autoencoder_optimizer.minimize(self.autoencoder_loss)

        self.discriminator_gaussian_optimizer = aae_helper.\
            get_optimizer(self, optimizer_discriminator, "discriminator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator)
        self.discriminator_gaussian_trainer = \
            self.discriminator_gaussian_optimizer.minimize(self.discriminator_gaussian_loss,
                                                           var_list=discriminator_gaussian_vars)

        self.discriminator_categorical_optimizer = aae_helper.\
            get_optimizer(self, optimizer_discriminator, "discriminator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator)
        self.discriminator_categorical_trainer = \
            self.discriminator_categorical_optimizer.minimize(self.discriminator_categorical_loss,
                                                              var_list=discriminator_categorical_vars)

        self.generator_optimizer = aae_helper.\
            get_optimizer(self, optimizer_generator, "generator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_generator)
        self.generator_trainer = \
            self.generator_optimizer.minimize(self.generator_loss, var_list=encoder_vars, global_step=self.global_step)

        # TODO: include in param dictionary
        # self.supervised_encoder_optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
        #                                                            beta1=0.9)
        self.supervised_encoder_optimizer = aae_helper.\
            get_optimizer(self, optimizer_autoencoder, "autoencoder", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.supervised_encoder_trainer = self.supervised_encoder_optimizer.minimize(self.supervised_encoder_loss,
                                                                                     var_list=encoder_vars)

        """
        Create the tensorboard summary
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=decoder_output, encoder_output=self.latent_variable_z,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_gaussian_loss=self.discriminator_gaussian_loss,
                                            discriminator_categorical_loss=self.discriminator_categorical_loss,
                                            generator_loss=self.generator_loss,
                                            supervised_encoder_loss=self.supervised_encoder_loss,
                                            real_distribution=self.real_distribution,
                                            encoder_output_label=self.class_label_y,
                                            categorical_distribution=self.categorial_distribution,
                                            decoder_output_multiple=self.decoder_output_multiple)

        """
        Variable for the "manual" summary
        """

        self.final_performance = None
        self.performance_over_time = {"autoencoder_losses": [], "discriminator_gaussian_losses": [],
                                      "discriminator_categorical_losses": [], "generator_losses": [],
                                      "supervised_encoder_loss": [], "accuracy": [], "accuracy_epochs": [],
                                      "list_of_epochs": []}
        self.learning_rates = {"autoencoder_lr": [], "discriminator_g_lr": [], "discriminator_c_lr": [],
                               "generator_lr": [], "supervised_encoder_lr": [], "list_of_epochs": []}

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    def get_final_performance(self):
        return self.final_performance

    def get_performance(self):
        return self.final_performance

    def get_result_folder_name(self):
        return self.result_folder_name

    def encoder(self, X, bias_init_values, reuse=False, is_supervised=False):
        """
        Encoder of the autoencoder.
        :param X: input to the autoencoder
        :param bias_init_values: the initial value for the bias
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :param is_supervised: True -> returns output without passing it through softmax,
                              False -> returns output after passing it through softmax.
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
                latent_variable_z = aae_helper.create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                                            bias_init_value=bias_init_values[0])

                categorical_encoder_label = aae_helper.create_dense_layer(latent_variable_z,
                                                                    self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                                    self.n_classes, 'encoder_label')

                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                latent_variable_z = aae_helper.create_dense_layer(dense_layer_1,
                                                            self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                            self.z_dim, 'encoder_output',
                                                            bias_init_value=bias_init_values[1])

                categorical_encoder_label = aae_helper.create_dense_layer(latent_variable_z,
                                                                    self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                                    self.n_classes, 'encoder_label')
                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'encoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(1, n_hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        aae_helper.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                'encoder_dense_layer_' + str(i + 1),
                                                bias_init_value=bias_init_values[i]))
                latent_variable_z = aae_helper.create_dense_layer(dense_layer_i,
                                                            self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                            self.z_dim, 'encoder_output',
                                                            bias_init_value=bias_init_values[-1])

                # label prediction of the encoder
                categorical_encoder_label = aae_helper.create_dense_layer(dense_layer_i,
                                                                    self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                                    self.n_classes, 'encoder_label')
                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

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
                    aae_helper.create_dense_layer(X, self.z_dim + self.n_classes, self.input_dim, 'decoder_output',
                                            bias_init_value=bias_init_values[0]))
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.z_dim + self.n_classes,
                                            self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            'decoder_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                decoder_output = tf.nn.sigmoid(
                    aae_helper.create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            self.input_dim,
                                            'decoder_output', bias_init_value=bias_init_values[1]))
                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.z_dim + self.n_classes,
                                            self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                            'decoder_dense_layer_1', bias_init_value=bias_init_values[0]))
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = tf.nn.relu(
                        aae_helper.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                                bias_init_value=bias_init_values[i]))
                decoder_output = tf.nn.sigmoid(
                    aae_helper.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                            self.input_dim,
                                            'decoder_output', bias_init_value=bias_init_values[-1]))
                return decoder_output

    def discriminator_gaussian(self, X, bias_init_values, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given gaussian prior distribution.
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
        with tf.name_scope('Discriminator_Gaussian'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = aae_helper.create_dense_layer(X, self.z_dim, 1, 'discriminator_gaussian_output',
                                                               bias_init_value=bias_init_values[0])
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_gaussian_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                discriminator_output = aae_helper.create_dense_layer(dense_layer_1,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                                               'discriminator_gaussian_output',
                                                               bias_init_value=bias_init_values[1])
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_gaussian_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        aae_helper.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                                self.n_neurons_of_hidden_layer_x_discriminator[i],
                                                'discriminator_gaussian_dense_layer_' +
                                                str(i + 1),
                                                bias_init_value=bias_init_values[i]))
                discriminator_output = aae_helper.create_dense_layer(dense_layer_i,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                                               'discriminator_gaussian_output',
                                                               bias_init_value=bias_init_values[-1])
                return discriminator_output

    def discriminator_categorical(self, X, bias_init_values, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given categorical prior distribution.
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
        with tf.name_scope('Discriminator_Categorical'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = aae_helper.create_dense_layer(X, self.n_classes, 1, 'discriminator_categorical_output',
                                                               bias_init_value=bias_init_values[0])
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.n_classes, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_categorical_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                discriminator_output = aae_helper.create_dense_layer(dense_layer_1,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                                               'discriminator_categorical_output',
                                                               bias_init_value=bias_init_values[1])
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = tf.nn.relu(
                    aae_helper.create_dense_layer(X, self.n_classes, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                            'discriminator_categorical_dense_layer_1',
                                            bias_init_value=bias_init_values[0]))
                for i in range(1, n__hidden_layers):
                    dense_layer_i = tf.nn.relu(
                        aae_helper.create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                                self.n_neurons_of_hidden_layer_x_discriminator[i],
                                                'discriminator_categorical_dense_layer_' +
                                                str(i + 1),
                                                bias_init_value=bias_init_values[i]))
                discriminator_output = aae_helper.create_dense_layer(dense_layer_i,
                                                               self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                                               'discriminator_categorical_output',
                                                               bias_init_value=bias_init_values[-1])
                return discriminator_output

    def get_mini_batch(self, x, y, batch_size):
        """
        Used to return a random batch from the given inputs.
        :param x: Input images of shape [None, 784]
        :param y: Input labels of shape [None, 10]
        :param batch_size: integer, batch size of images and labels to return
        :return: x -> [batch_size, 784], y-> [batch_size, 10]
        """
        index = np.arange(self.n_labeled)
        random_index = np.random.permutation(index)[:batch_size]
        return x[random_index], y[random_index]

    def create_tensorboard_summary(self, decoder_output, encoder_output, autoencoder_loss, discriminator_gaussian_loss,
                                   discriminator_categorical_loss, generator_loss, supervised_encoder_loss,
                                   real_distribution, encoder_output_label, categorical_distribution,
                                   decoder_output_multiple):
        """
        defines what should be shown in the tensorboard summary
        :param decoder_output:
        :param encoder_output:
        :param autoencoder_loss:
        :param discriminator_gaussian_loss:
        :param generator_loss:
        :param real_distribution:
        :return:
        """

        # Tensorboard visualization

        # Reshape images accordingly to the color scale to display them
        if self.color_scale == "rgb_scale":
            input_images = aae_helper.reshape_tensor_to_rgb_image(self.X, self.input_dim_x, self.input_dim_y)
            generated_images = aae_helper.reshape_tensor_to_rgb_image(decoder_output, self.input_dim_x, self.input_dim_y)
            generated_images_z_dist = aae_helper.reshape_tensor_to_rgb_image(decoder_output_multiple, self.input_dim_x,
                                                                                                        self.input_dim_y)
        else:
            input_images = tf.reshape(self.X, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images = tf.reshape(decoder_output, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images_z_dist = tf.reshape(decoder_output_multiple, [-1, self.input_dim_x, self.input_dim_y, 1])

        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Discriminator Gaussian Loss', tensor=discriminator_gaussian_loss)
        tf.summary.scalar(name='Discriminator Categorical Loss', tensor=discriminator_categorical_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.scalar(name='Supervised Encoder Loss', tensor=supervised_encoder_loss)
        tf.summary.histogram(name='Encoder Gaussian Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Gaussian Distribution', values=real_distribution)
        tf.summary.histogram(name='Encoder Categorical Distribution', values=encoder_output_label)
        tf.summary.histogram(name='Real Categorical Distribution', values=categorical_distribution)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=50)
        tf.summary.image(name='Generated Images from input images', tensor=generated_images, max_outputs=50)
        tf.summary.image(name='Generated Images z-dist', tensor=generated_images_z_dist, max_outputs=50)

        summary_op = tf.summary.merge_all()
        return summary_op

    def generate_image_grid(self, sess, op, epoch):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :return: None, displays a matplotlib window with all the merged images.
        """
        nx, ny = self.n_classes, self.n_classes
        random_inputs = np.random.randn(self.n_classes, self.z_dim) * 5.
        class_labels = np.identity(self.n_classes)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
        i = 0
        for r in random_inputs:
            for class_label_one_hot in class_labels:
                r, class_label_one_hot = np.reshape(r, (1, self.z_dim)), np.reshape(class_label_one_hot,
                                                                                    (1, self.n_classes))
                dec_input = np.concatenate((class_label_one_hot, r), 1)
                x = sess.run(op, feed_dict={self.decoder_input: dec_input})
                ax = plt.subplot(gs[i])
                i += 1

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
                    class_label = int(i / self.n_classes)
                    ax.set_ylabel(class_label, fontsize=9)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')
        # plt.show()

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        latent_representations_current_epoch = []
        labels_current_epoch = []

        # Get the data
        data = aae_helper.get_input_data(self.selected_dataset)

        # Saving the model
        saver = tf.train.Saver()
        # TODO: maybe worth a try..
        # saver = tf.train.Saver(tf.trainable_variables())

        autoencoder_loss_final, discriminator_loss_final, generator_loss_final, accuracy = 0, 0, 0, 0

        step = 0
        with tf.Session() as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:
                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = aae_helper.form_results(self)
                writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # write the parameter dictionary to some file
                json_dictionary = json.dumps(self.parameter_dictionary)
                with open(log_path + '/params.txt', 'a') as file:
                    file.write(json_dictionary)

                batch_x_labeled, batch_labels_labeled = data.train.next_batch(self.n_labeled)

                # we want n_epochs iterations
                for epoch in range(self.n_epochs):



                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(self.n_labeled / self.batch_size)

                    if self.verbose:
                        print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        z_real_dist = draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(self.batch_size, self.z_dim))
                        real_cat_dist = np.random.randint(low=0, high=10, size=self.batch_size)
                        real_cat_dist = np.eye(self.n_classes)[real_cat_dist]

                        # get the unlabeled batch from the training data
                        batch_X_unlabeled, batch_X_unlabeled_labels = data.train.next_batch(self.batch_size)

                        # get the labeled minibatch
                        mini_batch_X_labeled, mini_batch_labels = self.get_mini_batch(batch_x_labeled,
                                                                                      batch_labels_labeled,
                                                                                      batch_size=self.batch_size)

                        """
                        Reconstruction phase: autoencoder updates the encoder q(z, y|x) and the decoder to
                        minimize the reconstruction error of the inputs on an unlabeled mini-batch
                        """
                        # train the autoencoder by minimizing the reconstruction error between X and X_target
                        sess.run(self.autoencoder_trainer, feed_dict={self.X: batch_X_unlabeled,
                                                                        self.X_target: batch_X_unlabeled})

                        """
                        Regularization phase: each of the adversarial networks first updates their discriminative 
                        network to tell apart the true samples (generated using the Categorical and Gaussian priors) 
                        from the generated samples (the hidden codes computed by the autoencoder). The adversarial 
                        networks then update their generator to confuse their discriminative networks.
                        """
                        # train the discriminator to distinguish the true samples from the fake samples generated by the
                        # generator
                        sess.run(self.discriminator_gaussian_trainer,
                                 feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                            self.real_distribution: z_real_dist})
                        sess.run(self.discriminator_categorical_trainer,
                                 feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                            self.categorial_distribution: real_cat_dist})
                        # train the generator to fool the discriminator with its generated samples.
                        sess.run(self.generator_trainer, feed_dict={self.X: batch_X_unlabeled,
                                                                      self.X_target: batch_X_unlabeled})

                        """
                        Semi-supervised classification phase: autoencoder updates q(y|x) to minimize the cross-entropy 
                        cost on a labeled mini-batch.
                        """
                        # update encoder
                        sess.run(self.supervised_encoder_trainer,
                                 feed_dict={self.X_labeled: mini_batch_X_labeled, self.y: mini_batch_labels})

                        # every 5 epochs: write a summary for every 10th minibatch
                        if epoch % 5 == 0 and b % 10 == 0:

                            n_images_per_class = int(self.batch_size / self.n_classes)
                            class_labels_one_hot = np.identity(self.n_classes)
                            dec_inputs = []
                            for k in range(n_images_per_class):
                                for class_label in class_labels_one_hot:
                                    random_inputs = np.random.randn(1, self.z_dim) * 5.
                                    random_inputs = np.reshape(random_inputs, (1, self.z_dim))
                                    class_label = np.reshape(class_label, (1, self.n_classes))
                                    dec_inputs.append(np.concatenate((random_inputs, class_label), 1))
                            dec_inputs = np.array(dec_inputs).reshape(self.batch_size, self.z_dim + self.n_classes)

                            autoencoder_loss, discriminator_gaussian_loss, discriminator_categorical_loss, \
                            generator_loss, supervised_encoder_loss, summary, real_dist, \
                            latent_representation, encoder_cat_dist = sess.run(
                                [self.autoencoder_loss, self.discriminator_gaussian_loss,
                                 self.discriminator_categorical_loss, self.generator_loss, self.supervised_encoder_loss,
                                 self.tensorboard_summary, self.real_distribution, self.latent_variable_z,
                                 self.class_label_y],
                                feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                           self.real_distribution: z_real_dist, self.y: mini_batch_labels,
                                           self.X_labeled: mini_batch_X_labeled,
                                           self.categorial_distribution: real_cat_dist,
                                           self.decoder_input_multiple: dec_inputs})
                            writer.add_summary(summary, global_step=step)

                            latent_representations_current_epoch.extend(latent_representation)
                            labels_current_epoch.extend(batch_X_unlabeled_labels)


                            r = np.reshape(latent_representation[0, :], (1, self.z_dim))
                            class_label_one_hot = np.reshape(batch_X_unlabeled_labels[0, :], (1, self.n_classes))
                            dec_input = np.concatenate((class_label_one_hot, r), 1)

                            # dec_input = np.concatenate((batch_labels[0, :], latent_representation[0, :]), 1)
                            x = sess.run(self.decoder_output_real_dist, feed_dict={self.decoder_input: dec_input})

                            # 5 losses
                            # 5 learning rates
                            # real dist, encoder dist, input image, output image, accuracy
                            # encoder categorical, encoder gaussian, real categorical, real_gaussian

                            self.performance_over_time["autoencoder_losses"].append(autoencoder_loss)
                            self.performance_over_time["discriminator_gaussian_losses"].append(discriminator_gaussian_loss)
                            self.performance_over_time["discriminator_categorical_losses"].append(discriminator_categorical_loss)
                            self.performance_over_time["generator_losses"].append(generator_loss)
                            self.performance_over_time["supervised_encoder_loss"].append(supervised_encoder_loss)
                            self.performance_over_time["list_of_epochs"].append(epoch + (b / n_batches))

                            # update the dictionary holding the learning rates
                            self.learning_rates["autoencoder_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.autoencoder_optimizer)))
                            self.learning_rates["discriminator_g_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.discriminator_gaussian_optimizer)))
                            self.learning_rates["discriminator_c_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.discriminator_categorical_optimizer)))
                            self.learning_rates["generator_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.generator_optimizer)))
                            self.learning_rates["supervised_encoder_lr"].append(
                                sess.run(aae_helper.get_learning_rate_for_optimizer(self.supervised_encoder_optimizer)))
                            self.learning_rates["list_of_epochs"].append(epoch + (b / n_batches))

                            aae_helper.create_minibatch_summary_image_semi_supervised(self, real_dist, latent_representation, batch_X_unlabeled,
                                                            x, epoch, b, real_cat_dist, encoder_cat_dist)

                            # TODO: final loss for disc_gaussian and disc_categ and supervised_encoder_loss
                            autoencoder_loss_final = autoencoder_loss
                            discriminator_loss_final = discriminator_gaussian_loss + discriminator_categorical_loss
                            generator_loss_final = generator_loss

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(epoch, b))
                                print("Autoencoder Loss: {}".format(autoencoder_loss))
                                print("Discriminator Gauss Loss: {}".format(discriminator_gaussian_loss))
                                print("Discriminator Categorical Loss: {}".format(discriminator_categorical_loss))
                                print("Generator Loss: {}".format(generator_loss))
                                print("Supervised Loss: {}\n".format(supervised_encoder_loss))

                        step += 1

                    # evaluate the classification performance
                    accuracy = 0
                    num_batches = int(data.validation.num_examples / self.batch_size)
                    for j in range(num_batches):
                        # Classify unseen validation data instead of test data or train data
                        mini_batch_X_labeled, mini_batch_labels = data.validation.next_batch(batch_size=self.batch_size)
                        encoder_acc = sess.run(self.accuracy, feed_dict={self.X_labeled: mini_batch_X_labeled,
                                                                         self.y: mini_batch_labels})
                        accuracy += encoder_acc
                    accuracy /= num_batches
                    self.performance_over_time["accuracy"].append(accuracy)
                    self.performance_over_time["accuracy_epochs"].append(epoch)

                    if self.verbose:
                        print("Encoder Classification Accuracy: {}".format(accuracy))
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Encoder Classification Accuracy: {}\n".format(accuracy))

                    if epoch % 5 == 0:
                        self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=epoch)
                        plt.close('all')

                        if len(labels_current_epoch) > 0:
                            result_path = self.results_path + self.result_folder_name + '/Tensorboard/'
                            aae_helper.draw_class_distribution_on_latent_space(latent_representations_current_epoch,
                                                                               labels_current_epoch, result_path, epoch)

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
            self.end_training(autoencoder_loss_final, discriminator_loss_final, generator_loss_final, accuracy,
                              None, saver, sess, step)

    def end_training(self, autoencoder_loss_final, discriminator_loss_final, generator_loss_final, accuracy,
                     saved_model_path, saver, sess, step):
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
        if saved_model_path:
            saver.save(sess, save_path=saved_model_path, global_step=step)

        # print the final losses
        if self.verbose:
            print("Autoencoder Loss: {}".format(autoencoder_loss_final))
            print("Discriminator Loss: {}".format(discriminator_loss_final))
            print("Generator Loss: {}".format(generator_loss_final))
            print("Encoder Classification Accuracy: {}".format(accuracy))

        # set the final performance
        self.final_performance = {"autoencoder_loss_final": autoencoder_loss_final,
                                  "discriminator_loss_final": discriminator_loss_final,
                                  "generator_loss_final": generator_loss_final,
                                  "summed_loss_final": autoencoder_loss_final + discriminator_loss_final +
                                                 generator_loss_final,
                                  "accuracy": accuracy}
        # close the tensorflow session
        sess.close()
        # tf.reset_default_graph()