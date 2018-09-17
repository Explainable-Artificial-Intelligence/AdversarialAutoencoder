"""
    Implementation of a unsupervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
    https://arxiv.org/abs/1511.05644 by Goodfellow et. al. and the implementation available on
    https://github.com/Naresh1318/Adversarial_Autoencoder
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin

from swagger_server.utils.Storage import Storage
from util.AdversarialAutoencoderParameters import get_result_path_for_selected_autoencoder
from util.DataLoading import get_input_data
from util.Distributions import draw_from_single_gaussian
from util.NeuralNetworkUtils import get_loss_function, get_optimizer, get_layer_names, create_dense_layer, \
    form_results, get_learning_rate_for_optimizer, get_biases_or_weights_for_layer
from util.VisualizationUtils import reshape_tensor_to_rgb_image, reshape_image_array, create_epoch_summary_image, \
    create_reconstruction_grid, draw_class_distribution_on_latent_space, visualize_autoencoder_weights_and_biases, \
    create_gif, write_mass_spec_to_mgf_file, visualize_spectra_reconstruction, visualize_mass_spec_loss, \
    cluster_latent_space, reconstruct_generated_mass_spec_data, reconstruct_spectrum_from_feature_vector


class UnsupervisedAdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        # whether only the autoencoder and not the generative network should be trained
        if parameter_dictionary.get("only_train_autoencoder"):
            self.only_train_autoencoder = parameter_dictionary["only_train_autoencoder"]
        else:
            self.only_train_autoencoder = False

        # vars for the swagger server
        self.requested_operations_by_swagger = []
        self.requested_operations_by_swagger_results = None
        self.train_status = "start"

        self.result_folder_name = None
        self.parameter_dictionary = parameter_dictionary
        self.verbose = parameter_dictionary["verbose"]
        self.save_final_model = parameter_dictionary["save_final_model"]  # whether to save the final model
        self.write_tensorboard = parameter_dictionary["write_tensorboard"]  # whether to write the tensorboard file
        # create a summary image of the learning process every n epochs
        self.summary_image_frequency = parameter_dictionary["summary_image_frequency"]

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

        # dictionary holding some properties of the mass spec data; e.g. the organism name, the peak encoding,
        # the charge (if any) etc
        if parameter_dictionary.get("mass_spec_data_properties"):
            self.mass_spec_data_properties = parameter_dictionary["mass_spec_data_properties"]
        else:
            self.mass_spec_data_properties = None

        # load the data
        # TODO: noise and normalization as parameter
        data_properties = {"selected_dataset": self.selected_dataset, "color_scale": self.color_scale,
                           "data_normalized": False, "add_noise": False,
                           "mass_spec_data_properties": self.mass_spec_data_properties}

        # check if the data we need is already stored in the storage class..
        if not Storage.get_data_properties() == data_properties:
            # .. no; so we need to get it and store it in the storage class..
            data = get_input_data(self.selected_dataset, color_scale=self.color_scale, data_normalized=False,
                                  add_noise=False,
                                  mass_spec_data_properties=self.mass_spec_data_properties)
            Storage.set_selected_dataset(self.selected_dataset)
            Storage.set_data_properties(data_properties)
            Storage.set_input_data(data)

        """
        params for network topology
        """

        # number of neurons of the hidden layers
        self.n_neurons_of_hidden_layer_x_autoencoder = parameter_dictionary["n_neurons_of_hidden_layer_x_autoencoder"]
        self.n_neurons_of_hidden_layer_x_discriminator = \
            parameter_dictionary["n_neurons_of_hidden_layer_x_discriminator"]

        # dropout for the layers
        self.dropout_encoder = tf.placeholder_with_default([0.0] * len(parameter_dictionary["dropout_encoder"]),
                                                           shape=(len(parameter_dictionary["dropout_encoder"]),))
        self.dropout_decoder = tf.placeholder_with_default([0.0] * len(parameter_dictionary["dropout_decoder"]),
                                                           shape=(len(parameter_dictionary["dropout_decoder"]),))
        self.dropout_discriminator = \
            tf.placeholder_with_default([0.0] * len(parameter_dictionary["dropout_discriminator"]),
                                        shape=(len(parameter_dictionary["dropout_discriminator"]),))

        # what batch normalization to use for the different layers (no BN, post-activation, pre-activation)
        self.batch_normalization_encoder = parameter_dictionary["batch_normalization_encoder"]
        self.batch_normalization_decoder = parameter_dictionary["batch_normalization_decoder"]
        self.batch_normalization_discriminator = parameter_dictionary["batch_normalization_discriminator"]

        # convert "None" to None
        self.batch_normalization_encoder = [x if x is not "None" else None for x in self.batch_normalization_encoder]
        self.batch_normalization_decoder = [x if x is not "None" else None for x in self.batch_normalization_decoder]
        self.batch_normalization_discriminator = [x if x is not "None" else None for x in
                                                  self.batch_normalization_discriminator]

        # how the biases of the different layers should be initialized
        self.bias_initializer_encoder = parameter_dictionary["bias_initializer_encoder"]
        self.bias_initializer_decoder = parameter_dictionary["bias_initializer_decoder"]
        self.bias_initializer_discriminator = parameter_dictionary["bias_initializer_discriminator"]

        # parameters for the initialization of the different layers, e.g. mean and stddev for the
        # random_normal_initializer
        self.bias_initializer_params_encoder = parameter_dictionary["bias_initializer_params_encoder"]
        self.bias_initializer_params_decoder = parameter_dictionary["bias_initializer_params_decoder"]
        self.bias_initializer_params_discriminator = parameter_dictionary["bias_initializer_params_discriminator"]

        # how the weights of the different layers should be initialized
        self.weights_initializer_encoder = parameter_dictionary["weights_initializer_encoder"]
        self.weights_initializer_decoder = parameter_dictionary["weights_initializer_decoder"]
        self.weights_initializer_discriminator = parameter_dictionary["weights_initializer_discriminator"]

        # parameters for the initialization of the different layers, e.g. mean and stddev for the
        # random_normal_initializer
        self.weights_initializer_params_encoder = parameter_dictionary["weights_initializer_params_encoder"]
        self.weights_initializer_params_decoder = parameter_dictionary["weights_initializer_params_decoder"]
        self.weights_initializer_params_discriminator = parameter_dictionary["weights_initializer_params_discriminator"]

        # activation functions for the different parts of the network
        if type(parameter_dictionary["activation_function_encoder"]) is list:
            self.activation_function_encoder = parameter_dictionary["activation_function_encoder"]
        else:
            self.activation_function_encoder = [parameter_dictionary["activation_function_encoder"]] * \
                                               (len(self.n_neurons_of_hidden_layer_x_autoencoder) + 1)

        if type(parameter_dictionary["activation_function_decoder"]) is list:
            self.activation_function_decoder = parameter_dictionary["activation_function_decoder"]
        else:
            self.activation_function_decoder = [parameter_dictionary["activation_function_decoder"]] * \
                                               (len(self.n_neurons_of_hidden_layer_x_autoencoder) + 1)

        if type(parameter_dictionary["activation_function_discriminator"]) is list:
            self.activation_function_discriminator = parameter_dictionary["activation_function_discriminator"]
        else:
            self.activation_function_discriminator = [parameter_dictionary["activation_function_discriminator"]] * \
                                                     (len(self.n_neurons_of_hidden_layer_x_discriminator) + 1)

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
        self.increment_global_step_op = tf.assign_add(self.global_step, 1)

        # learning rate for the different parts of the network
        self.decaying_learning_rate_name_autoencoder = parameter_dictionary["decaying_learning_rate_name_autoencoder"]
        self.decaying_learning_rate_name_discriminator = \
            parameter_dictionary["decaying_learning_rate_name_discriminator"]
        self.decaying_learning_rate_name_generator = parameter_dictionary["decaying_learning_rate_name_generator"]

        """
        loss function
        """

        # loss function
        self.loss_function_discriminator = parameter_dictionary["loss_function_discriminator"]
        self.loss_function_generator = parameter_dictionary["loss_function_generator"]

        # path for the results
        self.results_path = parameter_dictionary["results_path"]
        if self.results_path is None:       # use default when no path is provided
            self.results_path = get_result_path_for_selected_autoencoder("Unsupervised")

        self.mz_loss_factor = parameter_dictionary.get("mz_loss_factor")
        self.intensity_loss_factor = parameter_dictionary.get("intensity_loss_factor")

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

        # for proper batch normalization
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        """
        Init the network; generator doesn't need to be initiated, since the generator is the encoder of the autoencoder
        """

        # init autoencoder
        with tf.variable_scope(tf.get_variable_scope()):
            # encoder part of the autoencoder and also the generator
            self.encoder_output = self.encoder(self.X)
            # decoder part of the autoencoder
            self.decoder_output = self.decoder(self.encoder_output)

        # init discriminator
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive samples p(z) (from a real data distribution)
            self.discriminator_pos_samples = \
                self.discriminator(self.real_distribution)
            # discriminator for the negative samples q(z) (generated by the generator)
            self.discriminator_neg_samples = \
                self.discriminator(self.encoder_output, reuse=True)

        # output of the decoder
        with tf.variable_scope(tf.get_variable_scope()):
            # used for "manually" passing single values drawn from a real distribution through the decoder for the
            # tensorboard
            self.decoder_output_real_dist = self.decoder(self.decoder_input, reuse=True)
            # used for "manually" passing multiple values drawn from a real distribution through the decoder for the
            # tensorboard
            self.decoder_output_multiple = \
                self.decoder(self.decoder_input_multiple, reuse=True)

        """
        Init the loss functions
        """
        # Autoencoder loss
        # for mass spec data we pick the reconstruction loss for the spectra; so we need to reconstruct the spectra
        # from the feature vector first
        if self.selected_dataset == "mass_spec":
            # original_images = self.X_target
            # reconstructed_images = self.decoder_output
            #
            # # reconstruct spectra from featurer vector
            # mz_values_original, intensities_original, charges_original, molecular_weights_original \
            #     = reconstruct_spectrum_from_feature_vector(original_images, self.input_dim,
            #                                                self.mass_spec_data_properties)
            # mz_values_reconstructed, intensities_reconstructed, charges_reconstructed, molecular_weights_reconstructed \
            #     = reconstruct_spectrum_from_feature_vector(reconstructed_images, self.input_dim,
            #                                                self.mass_spec_data_properties)
            #
            # # calculate the average difference between original and reconstruction
            # # self.mz_values_loss = tf.reduce_mean(tf.abs(mz_values_reconstructed - mz_values_original))
            # # self.intensities_loss = tf.reduce_mean(tf.abs(intensities_reconstructed - intensities_original))
            # self.mz_values_loss = tf.reduce_mean(tf.square(mz_values_reconstructed - mz_values_original))
            # self.intensities_loss = tf.reduce_mean(tf.square(intensities_reconstructed - intensities_original))
            # self.autoencoder_loss = self.mz_values_loss + self.intensities_loss

            # TODO: depending on peak encoding; parameter for factor of loss
            mz_values_original = self.X_target[:, ::2]
            mz_values_reconstructed = self.decoder_output[:, ::2]
            intensities_original = self.X_target[:, 1::2]
            intensities_reconstructed = self.decoder_output[:, 1::2]

            mz_values_loss = tf.square(mz_values_original - mz_values_reconstructed)
            intensities_loss = tf.square(intensities_original - intensities_reconstructed)

            self.autoencoder_loss = tf.reduce_mean(mz_values_loss * self.mz_loss_factor
                                                   + intensities_loss * self.intensity_loss_factor)

        # use the default autoencoder loss
        else:
            self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - self.decoder_output))
            #self.autoencoder_loss = tf.reduce_mean(tf.sqrt(tf.square(self.X_target - self.decoder_output)))

        # TODO: remove hard coded
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - self.decoder_output))

        # Discriminator Loss
        discriminator_loss_pos_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator,
                              labels=tf.ones_like(self.discriminator_pos_samples),
                              logits=self.discriminator_pos_samples))
        discriminator_loss_neg_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator,
                              labels=tf.zeros_like(self.discriminator_neg_samples),
                              logits=self.discriminator_neg_samples))
        self.discriminator_loss = discriminator_loss_neg_samples + discriminator_loss_pos_samples

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_generator,
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
        self.autoencoder_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_autoencoder, "autoencoder",
                          global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.autoencoder_trainer = self.autoencoder_optimizer.minimize(self.autoencoder_loss)
        self.discriminator_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_discriminator, "discriminator",
                          global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator)
        self.discriminator_trainer = self.discriminator_optimizer.minimize(self.discriminator_loss,
                                                                           var_list=discriminator_vars)
        self.generator_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_generator, "generator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_generator)
        self.generator_trainer = self.generator_optimizer.minimize(self.generator_loss, var_list=encoder_vars)

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

        # holds the tensorflow session
        self.session = tf.Session()

        """
        Variable for the "manual" summary
        """
        self.final_performance = None
        self.performance_over_time = {"autoencoder_losses": [], "discriminator_losses": [], "generator_losses": [],
                                      "list_of_epochs": [], "mz_values_losses": [], "intensities_losses": []}
        self.learning_rates = {"autoencoder_lr": [], "discriminator_lr": [], "generator_lr": [], "list_of_epochs": []}

        # variables for the minibatch summary image
        self.epoch_summary_vars = {"real_dist": [], "latent_representation": [], "discriminator_neg": [],
                                   "discriminator_pos": [], "batch_x": [], "reconstructed_images": [],
                                   "epoch": None, "batch_labels": []}

        # holds the original m/z and intensity values and their reconstruction (for the swagger server)
        self.spectra_original_and_reconstruction = {"mz_values_original": None, "mz_values_reconstructed": None,
                                                    "intensities_original": None, "intensities_reconstructed": None}

        # only for tuning; if set to true, the previous tuning results (losses and learning rates) are included in the
        # minibatch summary plots
        self.include_tuning_performance = False

        # holds the names for all layers
        self.all_layer_names = get_layer_names(self)

        self.latent_space_min_max_per_dim = []

        """
        Init all variables         
        """
        self.init = tf.global_variables_initializer()

    def get_all_layer_names(self):
        return self.all_layer_names

    def set_include_tuning_performance(self, include_tuning_performance):
        self.include_tuning_performance = include_tuning_performance

    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def get_requested_operations_by_swagger_results(self):
        return self.requested_operations_by_swagger_results

    def set_requested_operations_by_swagger_results(self, requested_operations_by_swagger_results):
        self.requested_operations_by_swagger_results = requested_operations_by_swagger_results

    def get_requested_operations_by_swagger(self):
        return self.requested_operations_by_swagger

    def add_to_requested_operations_by_swagger(self, requested_operation):
        self.requested_operations_by_swagger.append(requested_operation)

    def get_epoch_summary_vars(self):
        return self.epoch_summary_vars

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

    def set_result_folder_name(self, result_folder_name):
        self.result_folder_name = result_folder_name

    def get_result_folder_name(self):
        return self.result_folder_name

    def set_spectra_original_and_reconstruction(self, mz_values_original, mz_values_reconstructed,
                                                intensities_original, intensities_reconstructed):
        self.spectra_original_and_reconstruction = {"mz_values_original": mz_values_original,
                                                    "mz_values_reconstructed": mz_values_reconstructed,
                                                    "intensities_original": intensities_original,
                                                    "intensities_reconstructed": intensities_reconstructed}

    def get_spectra_original_and_reconstruction(self):
        return self.spectra_original_and_reconstruction

    def encoder(self, X, reuse=False):
        """
        Encoder of the autoencoder.
        :param X: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """

        # number of hidden layers
        n_hidden_layers = len(self.n_neurons_of_hidden_layer_x_autoencoder)

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Encoder'
        with tf.name_scope('Encoder'):
            # there is no hidden layer
            if n_hidden_layers == 0:

                last_layer = \
                    create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                       weight_initializer=self.weights_initializer_encoder[0],
                                       weight_initializer_params=self.weights_initializer_params_encoder[0],
                                       bias_initializer=self.bias_initializer_encoder[0],
                                       bias_initializer_params=self.bias_initializer_params_encoder[0],
                                       activation_function=self.activation_function_encoder[0],
                                       batch_normalization=self.batch_normalization_encoder[0],
                                       drop_out_rate_input_layer=self.dropout_encoder[0],
                                       drop_out_rate_output_layer=self.dropout_encoder[1],
                                       is_training=self.is_training)
                return last_layer
            # there is only one hidden layer
            elif n_hidden_layers == 1:

                dense_layer_1 = \
                    create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       'encoder_dense_layer_1',
                                       weight_initializer=self.weights_initializer_encoder[0],
                                       weight_initializer_params=self.weights_initializer_params_encoder[0],
                                       bias_initializer=self.bias_initializer_encoder[0],
                                       bias_initializer_params=self.bias_initializer_params_encoder[0],
                                       activation_function=self.activation_function_encoder[0],
                                       batch_normalization=self.batch_normalization_encoder[0],
                                       drop_out_rate_input_layer=self.dropout_encoder[0],
                                       drop_out_rate_output_layer=self.dropout_encoder[1],
                                       is_training=self.is_training)

                last_layer = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       self.z_dim, 'encoder_output',
                                       weight_initializer=self.weights_initializer_encoder[1],
                                       weight_initializer_params=self.weights_initializer_params_encoder[1],
                                       bias_initializer=self.bias_initializer_encoder[1],
                                       bias_initializer_params=self.bias_initializer_params_encoder[1],
                                       activation_function=self.activation_function_encoder[1],
                                       batch_normalization=self.batch_normalization_encoder[1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_encoder[2],
                                       is_training=self.is_training)

                return last_layer
            # there is an arbitrary number of hidden layers
            else:

                dense_layer_i = \
                    create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       'encoder_dense_layer_1',
                                       weight_initializer=self.weights_initializer_encoder[0],
                                       weight_initializer_params=self.weights_initializer_params_encoder[0],
                                       bias_initializer=self.bias_initializer_encoder[0],
                                       bias_initializer_params=self.bias_initializer_params_encoder[0],
                                       activation_function=self.activation_function_encoder[0],
                                       batch_normalization=self.batch_normalization_encoder[0],
                                       drop_out_rate_input_layer=self.dropout_encoder[0],
                                       drop_out_rate_output_layer=self.dropout_encoder[1],
                                       is_training=self.is_training)

                for i in range(1, n_hidden_layers):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i,
                                           self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                           self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                           'encoder_dense_layer_' + str(i + 1),
                                           weight_initializer=self.weights_initializer_encoder[i],
                                           weight_initializer_params=self.weights_initializer_params_encoder[
                                               i],
                                           bias_initializer=self.bias_initializer_encoder[i],
                                           bias_initializer_params=self.bias_initializer_params_encoder[i],
                                           activation_function=self.activation_function_encoder[i],
                                           batch_normalization=self.batch_normalization_encoder[i],
                                           drop_out_rate_input_layer=0.0,
                                           drop_out_rate_output_layer=self.dropout_encoder[i + 1],
                                           is_training=self.is_training)

                last_layer = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                       self.z_dim, 'encoder_output',
                                       weight_initializer=self.weights_initializer_encoder[-1],
                                       weight_initializer_params=self.weights_initializer_params_encoder[-1],
                                       bias_initializer=self.bias_initializer_encoder[-1],
                                       bias_initializer_params=self.bias_initializer_params_encoder[-1],
                                       activation_function=self.activation_function_encoder[-1],
                                       batch_normalization=self.batch_normalization_encoder[-1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_encoder[-1],
                                       is_training=self.is_training)
                return last_layer

    def decoder(self, X, reuse=False):
        """
        Decoder of the autoencoder.
        :param X: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        # number of hidden layers
        n_hidden_layers = len(self.n_neurons_of_hidden_layer_x_autoencoder)

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Decoder'
        with tf.name_scope('Decoder'):
            # there is no hidden layer
            if n_hidden_layers == 0:

                decoder_output = \
                    create_dense_layer(X, self.z_dim, self.input_dim, 'x_reconstructed',
                                       weight_initializer=self.weights_initializer_decoder[0],
                                       weight_initializer_params=self.weights_initializer_params_decoder[0],
                                       bias_initializer=self.bias_initializer_decoder[0],
                                       bias_initializer_params=self.bias_initializer_params_decoder[0],
                                       activation_function=self.activation_function_decoder[0],
                                       batch_normalization=self.batch_normalization_decoder[0],
                                       drop_out_rate_input_layer=self.dropout_decoder[0],
                                       drop_out_rate_output_layer=self.dropout_decoder[1],
                                       is_training=self.is_training)
                return decoder_output
            # there is only one hidden layer
            elif n_hidden_layers == 1:

                dense_layer_1 = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       'decoder_dense_layer_1',
                                       weight_initializer=self.weights_initializer_decoder[0],
                                       weight_initializer_params=self.weights_initializer_params_decoder[0],
                                       bias_initializer=self.bias_initializer_decoder[0],
                                       bias_initializer_params=self.bias_initializer_params_decoder[0],
                                       activation_function=self.activation_function_decoder[0],
                                       batch_normalization=self.batch_normalization_decoder[0],
                                       drop_out_rate_input_layer=self.dropout_decoder[0],
                                       drop_out_rate_output_layer=self.dropout_decoder[1],
                                       is_training=self.is_training)

                decoder_output = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       self.input_dim, 'x_reconstructed',
                                       weight_initializer=self.weights_initializer_decoder[1],
                                       weight_initializer_params=self.weights_initializer_params_decoder[1],
                                       bias_initializer=self.bias_initializer_decoder[-1],
                                       bias_initializer_params=self.bias_initializer_params_decoder[1],
                                       activation_function=self.activation_function_decoder[1],
                                       batch_normalization=self.batch_normalization_decoder[1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_decoder[2],
                                       is_training=self.is_training)

                return decoder_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                       'decoder_dense_layer_1',
                                       weight_initializer=self.weights_initializer_decoder[0],
                                       weight_initializer_params=self.weights_initializer_params_decoder[0],
                                       bias_initializer=self.bias_initializer_decoder[0],
                                       bias_initializer_params=self.bias_initializer_params_decoder[0],
                                       activation_function=self.activation_function_decoder[0],
                                       batch_normalization=self.batch_normalization_decoder[0],
                                       drop_out_rate_input_layer=self.dropout_decoder[0],
                                       drop_out_rate_output_layer=self.dropout_decoder[1],
                                       is_training=self.is_training)
                for i in range(n_hidden_layers - 1, 0, -1):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                           self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                           'decoder_dense_layer_' + str(n_hidden_layers - i + 1),
                                           weight_initializer=self.weights_initializer_decoder[i],
                                           weight_initializer_params=self.weights_initializer_params_decoder[
                                               i],
                                           bias_initializer=self.bias_initializer_decoder[i],
                                           bias_initializer_params=self.bias_initializer_params_decoder[i],
                                           activation_function=self.activation_function_decoder[i],
                                           batch_normalization=self.batch_normalization_decoder[i],
                                           drop_out_rate_input_layer=0.0,
                                           drop_out_rate_output_layer=self.dropout_decoder[n_hidden_layers - i + 1],
                                           is_training=self.is_training)
                decoder_output = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                       self.input_dim, 'x_reconstructed',
                                       weight_initializer=self.weights_initializer_decoder[-1],
                                       weight_initializer_params=self.weights_initializer_params_decoder[-1],
                                       bias_initializer=self.bias_initializer_decoder[-1],
                                       bias_initializer_params=self.bias_initializer_params_decoder[-1],
                                       activation_function=self.activation_function_decoder[-1],
                                       batch_normalization=self.batch_normalization_decoder[-1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_decoder[-1],
                                       is_training=self.is_training)
                return decoder_output

    def discriminator(self, X, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param X: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """

        # number of hidden layers
        n__hidden_layers = len(self.n_neurons_of_hidden_layer_x_discriminator)

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Discriminator'
        with tf.name_scope('Discriminator'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = \
                    create_dense_layer(X, self.z_dim, 1, 'discriminator_output',
                                       weight_initializer=self.weights_initializer_discriminator[0],
                                       weight_initializer_params=self.weights_initializer_params_discriminator[0],
                                       bias_initializer=self.bias_initializer_discriminator[0],
                                       bias_initializer_params=self.bias_initializer_params_discriminator[0],
                                       activation_function=self.activation_function_discriminator[0],
                                       batch_normalization=self.batch_normalization_discriminator[0],
                                       drop_out_rate_input_layer=self.dropout_discriminator[0],
                                       drop_out_rate_output_layer=self.dropout_discriminator[1],
                                       is_training=self.is_training)
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                       'discriminator_dense_layer_1',
                                       weight_initializer=self.weights_initializer_discriminator[0],
                                       weight_initializer_params=
                                       self.weights_initializer_params_discriminator[0],
                                       bias_initializer=self.bias_initializer_discriminator[0],
                                       bias_initializer_params=self.bias_initializer_params_discriminator[0],
                                       activation_function=self.activation_function_discriminator[0],
                                       batch_normalization=self.batch_normalization_discriminator[0],
                                       drop_out_rate_input_layer=self.dropout_discriminator[0],
                                       drop_out_rate_output_layer=self.dropout_discriminator[1],
                                       is_training=self.is_training)
                discriminator_output = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator[0], 1,
                                       'discriminator_output',
                                       weight_initializer=self.weights_initializer_discriminator[-1],
                                       weight_initializer_params=
                                       self.weights_initializer_params_discriminator[-1],
                                       bias_initializer=self.bias_initializer_discriminator[-1],
                                       bias_initializer_params=self.bias_initializer_params_discriminator[-1],
                                       activation_function=self.activation_function_discriminator[-1],
                                       batch_normalization=self.batch_normalization_discriminator[-1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_discriminator[-1],
                                       is_training=self.is_training)
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator[0],
                                       'discriminator_dense_layer_1',
                                       weight_initializer=self.weights_initializer_discriminator[0],
                                       weight_initializer_params=
                                       self.weights_initializer_params_discriminator[0],
                                       bias_initializer=self.bias_initializer_discriminator[0],
                                       bias_initializer_params=self.bias_initializer_params_discriminator[0],
                                       activation_function=self.activation_function_discriminator[0],
                                       batch_normalization=self.batch_normalization_discriminator[0],
                                       drop_out_rate_input_layer=self.dropout_discriminator[0],
                                       drop_out_rate_output_layer=self.dropout_discriminator[1],
                                       is_training=self.is_training)
                for i in range(1, n__hidden_layers):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i,
                                           self.n_neurons_of_hidden_layer_x_discriminator[i - 1],
                                           self.n_neurons_of_hidden_layer_x_discriminator[i],
                                           'discriminator_dense_layer_' + str(i + 1),
                                           weight_initializer=self.weights_initializer_discriminator[i],
                                           weight_initializer_params=
                                           self.weights_initializer_params_discriminator[i],
                                           bias_initializer=self.bias_initializer_discriminator[i],
                                           bias_initializer_params=
                                           self.bias_initializer_params_discriminator[i],
                                           activation_function=self.activation_function_discriminator[i],
                                           batch_normalization=self.batch_normalization_discriminator[i],
                                           drop_out_rate_input_layer=0.0,
                                           drop_out_rate_output_layer=self.dropout_discriminator[i + 1],
                                           is_training=self.is_training)

                discriminator_output = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator[-1], 1,
                                       'discriminator_output',
                                       weight_initializer=self.weights_initializer_discriminator[-1],
                                       weight_initializer_params=
                                       self.weights_initializer_params_discriminator[-1],
                                       bias_initializer=self.bias_initializer_discriminator[-1],
                                       bias_initializer_params=self.bias_initializer_params_discriminator[-1],
                                       activation_function=self.activation_function_discriminator[-1],
                                       batch_normalization=self.batch_normalization_discriminator[-1],
                                       drop_out_rate_input_layer=0.0,
                                       drop_out_rate_output_layer=self.dropout_discriminator[-1],
                                       is_training=self.is_training)
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
            input_images = reshape_tensor_to_rgb_image(self.X, self.input_dim_x, self.input_dim_y)
            generated_images = reshape_tensor_to_rgb_image(decoder_output, self.input_dim_x,
                                                           self.input_dim_y)
            generated_images_z_dist = reshape_tensor_to_rgb_image(decoder_output_multiple,
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

    def generate_image_grid(self, sess, op, epoch, points=None, left_cell=None, save_image_grid=True):
        """
        Generates and saves a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param points: optional; array of points on the latent space which should be used to generate the images
        :param left_cell: left cell of the grid spec with two adjacent horizontal cells holding the image grid
        and the class distribution on the latent space; if left_cell is None, then only the image grid is supposed
        to be plotted and not the "combinated" image (image grid + class distr. on latent space).
        :param save_image_grid: whether to save the image grid
        :return:
        """

        if self.z_dim > 2:
            # randomly sample some points from the z dim space
            image_grid_x_length = 20
            image_grid_y_length = 20
            n_points_to_sample = image_grid_x_length * image_grid_y_length

            if not points:
                # randomly sample some points from the z dim space
                points = np.random.uniform(-10, 10, [n_points_to_sample, self.z_dim])

        else:
            # creates evenly spaced values within [-10, 10] with a spacing of 1.5
            x_points = np.arange(15, -15, -1.5).astype(np.float32)
            y_points = np.arange(-15, 15, 1.5).astype(np.float32)

        nx, ny = 20, 20
        # create the image grid
        if left_cell:
            gs = gridspec.GridSpecFromSubplotSpec(nx, ny, left_cell)
        else:
            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        list_of_images_for_swagger = []
        counter_label_x_axis = 0
        counter_label_y_axis = 0

        # iterate over the image grid
        for i, g in enumerate(gs):

            # # create a data point from the random_points array
            if self.z_dim > 2:
                z = points[i]
                # z = random_points[i]
                # print(z)
            else:
                # create a data point from the x_points and y_points array as input for the decoder
                z = np.concatenate(([y_points[int(i % nx)]], [x_points[int(i / ny)]]))

            z = np.reshape(z, (1, self.z_dim))

            # run the decoder
            x = sess.run(op, feed_dict={self.decoder_input: z, self.is_training: False})
            x = np.array(x).reshape(self.input_dim)

            ax = plt.subplot(g)

            # reshape the image array and display it
            img = reshape_image_array(self, x)
            list_of_images_for_swagger.append(img)
            if self.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            if self.z_dim == 2:
                # create the label for the y axis
                if ax.is_first_col():
                    if counter_label_y_axis % 2 == 0:
                        ax.set_ylabel(x_points[int(i / ny)], fontsize=9)
                    counter_label_y_axis += 1

                # create the label x for the x axis
                if ax.is_last_row():
                    if counter_label_x_axis % 2 == 0:
                        ax.set_xlabel(y_points[int(i % ny)], fontsize=9)
                    counter_label_x_axis += 1

        if not left_cell and save_image_grid:
            # save the created image grid
            plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

        return list_of_images_for_swagger

    def generate_image_from_single_point(self, sess, single_point):
        """
        generates a image from a single point in the latent space
        :param sess: tensorflow session
        :param single_point: coordinates of the point in latent space
        :return: np array representing the generated image
        """

        # reshape the point
        single_point = np.reshape(single_point, (1, self.z_dim))

        # run the point through the decoder to generate the image
        generated_image = sess.run(self.decoder_output_real_dist, feed_dict={self.decoder_input: single_point,
                                                                             self.is_training: False})

        # reshape the image array and display it
        generated_image = np.array(generated_image).reshape(self.input_dim)
        img = reshape_image_array(self, generated_image)

        if self.color_scale == "gray_scale":
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + "test_algorithm" + '.png')

        # if we have mass spec data, we need to reconstruct the data first
        if self.selected_dataset == "mass_spec":
            mz_values, intensities, charges, molecular_weights = \
                reconstruct_spectrum_from_feature_vector(img, self.input_dim, self.mass_spec_data_properties)
            # convert the numpy arrays to lists
            if isinstance(mz_values, np.ndarray):
                mz_values = mz_values.tolist()
            if isinstance(intensities, np.ndarray):
                intensities = intensities.tolist()
            if isinstance(charges, np.ndarray):
                charges = charges.tolist()
            if isinstance(molecular_weights, np.ndarray):
                molecular_weights = molecular_weights.tolist()
            return [mz_values, intensities, charges, molecular_weights]

        return img

    def train(self, is_train_mode_active=True):
        """
        trains the adversarial autoencoder on the MNIST data set or generates the image grid using the previously
        trained model
        :param is_train_mode_active: whether a autoencoder should be trained or not
        :return:
        """

        # we need a new session, if training has been completed and the old session has been closed
        if not is_train_mode_active:
            self.session = tf.Session()

        saved_model_path = None
        log_path = None
        writer = None

        latent_representations_current_epoch = []
        labels_current_epoch = []

        # get the data from the storage class
        data = Storage.get_all_input_data()

        autoencoder_epoch_losses, discriminator_epoch_losses, generator_epoch_losses = [], [], []
        epochs_completed = 0
        step = 0
        with self.session as sess:

            # init the tf variables
            sess.run(self.init)

            # train the autoencoder
            if is_train_mode_active:

                # creates folders for each run to store the tensorboard files, saved models and the log files.
                tensorboard_path, saved_model_path, log_path = form_results(self)
                if self.write_tensorboard:
                    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

                # write the used parameter dictionary to some file
                json_dictionary = json.dumps(self.parameter_dictionary)
                with open(log_path + '/params.txt', 'w') as file:
                    file.write(json_dictionary)

                # visualize the weights and biases before training
                visualize_autoencoder_weights_and_biases(self, epoch="before_training")

                # we want n_epochs iterations
                for epoch in range(self.n_epochs):

                    if self.train_status == "stop" and epoch > 0:
                        # end the training
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

                        self.process_requested_swagger_operations(sess)

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        # z_real_dist = draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(self.batch_size, self.z_dim))
                        z_real_dist = \
                            draw_from_single_gaussian(mean=0.0, std_dev=1.0, shape=(self.batch_size, self.z_dim)) * 5
                        # z_real_dist = draw_from_dim_reduced_dataset(self.batch_size, self.selected_dataset, self.z_dim)

                        # get the batch from the training data
                        batch_x, batch_labels = data.train.next_batch(self.batch_size)
                        # batch_x, batch_labels = data.train.get_class_specific_batch(self.batch_size, 1)
                        # batch_x, batch_labels = \
                        #     data.train.get_color_specific_image_combinations(self.batch_size, 0, 5, 1, self.input_dim_x,
                        #                                                      self.input_dim_y)

                        """
                        Reconstruction phase: the autoencoder updates the encoder and the decoder to minimize the
                        reconstruction error of the inputs
                        """
                        # train the autoencoder by minimizing the reconstruction error
                        sess.run(self.autoencoder_trainer,
                                 feed_dict={self.X: batch_x, self.X_target: batch_x,
                                            self.is_training: True,
                                            self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                            self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                            self.dropout_discriminator:
                                                self.parameter_dictionary["dropout_discriminator"]})

                        """
                        Regularization phase: the adversarial network first updates its discriminative network
                        to tell apart the true samples (generated using the prior) from the generated samples (the 
                        hidden codes computed by the autoencoder). The adversarial network then updates its generator 
                        (which is also the encoder of the autoencoder) to confuse the discriminative network.
                        """
                        if not self.only_train_autoencoder:
                            # train the discriminator to distinguish the true samples from the fake samples generated
                            # by the generator
                            sess.run(self.discriminator_trainer,
                                     feed_dict={self.X: batch_x, self.X_target: batch_x,
                                                self.real_distribution: z_real_dist,
                                                self.is_training: True,
                                                self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                                self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                                self.dropout_discriminator:
                                                    self.parameter_dictionary["dropout_discriminator"]})
                            # train the generator to fool the discriminator with its generated samples.
                            sess.run(self.generator_trainer,
                                     feed_dict={self.X: batch_x, self.X_target: batch_x,
                                                self.is_training: True,
                                                self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                                self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                                self.dropout_discriminator:
                                                    self.parameter_dictionary["dropout_discriminator"]})

                        # every x epochs: write a summary for every 50th minibatch
                        if epoch % self.summary_image_frequency == 0 and b % 50 == 0:

                            autoencoder_loss, discriminator_loss, generator_loss, summary, real_dist, \
                            latent_representation, discriminator_neg, discriminator_pos, decoder_output = \
                                sess.run(
                                    [self.autoencoder_loss, self.discriminator_loss, self.generator_loss,
                                     self.tensorboard_summary, self.real_distribution, self.encoder_output,
                                     self.discriminator_neg_samples, self.discriminator_pos_samples,
                                     self.decoder_output],
                                    feed_dict={self.X: batch_x, self.X_target: batch_x,
                                               self.is_training: False,
                                               self.real_distribution: z_real_dist,
                                               self.decoder_input_multiple: z_real_dist})
                            if self.write_tensorboard:
                                writer.add_summary(summary, global_step=step)

                            latent_representations_current_epoch.extend(latent_representation)
                            labels_current_epoch.extend(batch_labels)

                            # update the lists holding the losses for each epoch
                            autoencoder_epoch_losses.append(autoencoder_loss)
                            discriminator_epoch_losses.append(discriminator_loss)
                            generator_epoch_losses.append(generator_loss)

                            # update the dictionary holding the learning rates
                            self.learning_rates["autoencoder_lr"].append(
                                get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess))
                            self.learning_rates["discriminator_lr"].append(
                                get_learning_rate_for_optimizer(self.discriminator_optimizer, sess))
                            self.learning_rates["generator_lr"].append(
                                get_learning_rate_for_optimizer(self.generator_optimizer, sess))
                            self.learning_rates["list_of_epochs"].append(epoch + (b / n_batches))

                            # updates vars for the swagger server
                            self.epoch_summary_vars["real_dist"].extend(real_dist)
                            self.epoch_summary_vars["latent_representation"].extend(latent_representation)
                            self.epoch_summary_vars["discriminator_neg"].extend(discriminator_neg)
                            self.epoch_summary_vars["discriminator_pos"].extend(discriminator_pos)
                            self.epoch_summary_vars["batch_x"].extend(batch_x)
                            self.epoch_summary_vars["reconstructed_images"].extend(decoder_output)
                            self.epoch_summary_vars["epoch"] = epoch
                            self.epoch_summary_vars["batch_labels"].extend(batch_labels)

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(epoch, b))
                                print("summed losses:", autoencoder_loss + discriminator_loss + generator_loss)
                                print("Autoencoder Loss: {}".format(autoencoder_loss))
                                print("Discriminator Loss: {}".format(discriminator_loss))
                                print("Generator Loss: {}".format(generator_loss))
                                print('Learning rate autoencoder: {}'.format(
                                    get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess)))
                                print('Learning rate discriminator: {}'.format(
                                    get_learning_rate_for_optimizer(self.discriminator_optimizer, sess)))
                                print('Learning rate generator: {}'.format(
                                    get_learning_rate_for_optimizer(self.generator_optimizer, sess)))

                            autoencoder_loss_final = autoencoder_loss
                            discriminator_loss_final = discriminator_loss
                            generator_loss_final = generator_loss

                            with open(log_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                                log.write("Autoencoder Loss: {}\n".format(autoencoder_loss))
                                log.write("Discriminator Loss: {}\n".format(discriminator_loss))
                                log.write("Generator Loss: {}\n".format(generator_loss))

                        step += 1

                    # increment the global step:
                    sess.run(self.increment_global_step_op)
                    epochs_completed += 1

                    # every x epochs ..
                    if epoch % self.summary_image_frequency == 0:

                        # update the lists holding the losses
                        self.performance_over_time["autoencoder_losses"].append(np.mean(autoencoder_epoch_losses))
                        self.performance_over_time["discriminator_losses"].append(np.mean(discriminator_epoch_losses))
                        self.performance_over_time["generator_losses"].append(np.mean(generator_epoch_losses))
                        self.performance_over_time["list_of_epochs"].append(epoch)

                        autoencoder_epoch_losses, discriminator_epoch_losses, generator_epoch_losses = [], [], []

                        # create the summary image for the current minibatch
                        create_epoch_summary_image(self, epoch, self.include_tuning_performance)

                        real_images = np.array(self.epoch_summary_vars["batch_x"])
                        reconstructed_images = np.array(self.epoch_summary_vars["reconstructed_images"])
                        create_reconstruction_grid(self, real_images, reconstructed_images, epoch=epoch)

                        if self.selected_dataset == "mass_spec":
                            write_mass_spec_to_mgf_file(self, epoch, reconstructed_images, real_images)
                            mz_values_loss, intensities_loss = visualize_spectra_reconstruction(self, epoch,
                                                                                                reconstructed_images,
                                                                                                real_images)
                            # update the lists holding the losses
                            self.performance_over_time["mz_values_losses"].append(np.mean(mz_values_loss))
                            self.performance_over_time["intensities_losses"].append(np.mean(intensities_loss))

                            with open(log_path + '/log_mass_spec_reconstruction.txt', 'a') as log:
                                log.write("Epoch: {}\n".format(epoch))
                                log.write("M/Z values Loss: {}\n".format(np.mean(mz_values_loss)))
                                log.write("Intensities Loss: {}\n".format(np.mean(intensities_loss)))

                            # visualize the mass spec loss
                            visualize_mass_spec_loss(self, epoch)

                        # increase figure size
                        plt.rcParams["figure.figsize"] = (6.4 * 2, 4.8)
                        outer_grid = gridspec.GridSpec(1, 2)
                        left_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid

                        # generate the image grid for the latent space
                        generated_images = self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=epoch,
                                                                    left_cell=left_cell,
                                                                    # points=latent_representations_current_epoch)
                        points = None)      # TODO: previous points=latent_representations_current_epoch
                        # TODO:





                        if self.selected_dataset == "mass_spec":
                            reconstruct_generated_mass_spec_data(self, generated_mass_spec_data=generated_images,
                                                                 epoch=epoch)

                        # draw the class distribution on the latent space
                        result_path = self.results_path + self.result_folder_name + '/Tensorboard/'
                        draw_class_distribution_on_latent_space(latent_representations_current_epoch,
                                                                labels_current_epoch, result_path, epoch,
                                                                None, combined_plot=True)

                        # cluster the latent space
                        cluster_latent_space(latent_representations_current_epoch, labels_current_epoch, result_path,
                                             epoch)

                        """
                        Weights + biases visualization
                        """
                        visualize_autoencoder_weights_and_biases(self, epoch=epoch)

                    # reset the list holding the latent representations for the current epoch
                    self.epoch_summary_vars = {"real_dist": [], "latent_representation": [], "discriminator_neg": [],
                                               "discriminator_pos": [], "batch_x": [], "reconstructed_images": [],
                                               "epoch": None, "batch_labels": []}
                    latent_representations_current_epoch = []
                    labels_current_epoch = []

            # display the generated images of the latest trained autoencoder
            else:
                # Get the latest results folder
                all_results = os.listdir(self.results_path)
                all_results.sort()

                self.saver.restore(sess, save_path=tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1]
                                                                              + '/Saved_models/'))

                self.process_requested_swagger_operations(sess)

                self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch="last", points=None)

            # after training..
            if epochs_completed > 0:
                # end the training
                self.end_training(saved_model_path, log_path, self.saver, sess, step)

    def process_requested_swagger_operations(self, sess):
        """
        processes the operations requested by swagger
        :param sess: tensorflow session
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
                        result = self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=None,
                                                          left_cell=None, save_image_grid=False, points=None)
                        self.set_requested_operations_by_swagger_results(result)
                    elif function_name == "generate_image_from_single_point":
                        result = self.generate_image_from_single_point(sess, function_params)
                        self.set_requested_operations_by_swagger_results(result)
                    elif function_name == "get_biases_or_weights_for_layer":
                        result = get_biases_or_weights_for_layer(self, function_params)
                        self.set_requested_operations_by_swagger_results(result)

                    plt.close('all')

            # reset the list
            self.requested_operations_by_swagger = []

    def end_training(self, saved_model_path, log_path, saver, sess, step):
        """
        ends the training by saving the model if a model path is provided, validating on the test dataset and closing
        the tf session
        :param saved_model_path: path where the saved model should be stored
        :param log_path: path where the log should be written to
        :param saver: tf.train.Saver() to save the model
        :param sess: session to save and to close
        :param step: global step
        :return:
        """
        # save the session if a path for the saved model is provided
        if saved_model_path and self.save_final_model:
            saver.save(sess, save_path=saved_model_path, global_step=step)

        # validate on test data
        test_data = Storage.get_input_data("test")
        n_batches = int(test_data.num_examples / self.batch_size)

        # lists holding the loss per batch
        autoencoder_loss_batch_list = []
        discriminator_loss_batch_list = []
        generator_loss_batch_list = []
        mz_values_loss_batch_list = []
        intensities_loss_batch_list = []

        # iterate over the batches and calculate the loss
        for b in range(n_batches):
            batch_x, batch_labels = test_data.next_batch(self.batch_size)

            z_real_dist = \
                draw_from_single_gaussian(mean=0.0, std_dev=1.0, shape=(self.batch_size, self.z_dim)) * 5

            autoencoder_loss, discriminator_loss, generator_loss, decoder_output = \
                sess.run(
                    [self.autoencoder_loss, self.discriminator_loss, self.generator_loss, self.decoder_output],
                    feed_dict={self.X: batch_x, self.X_target: batch_x,
                               self.is_training: False,
                               self.real_distribution: z_real_dist,
                               self.decoder_input_multiple: z_real_dist})

            real_images = batch_x
            reconstructed_images = decoder_output

            if self.selected_dataset == "mass_spec":
                mz_values_loss, intensities_loss \
                    = visualize_spectra_reconstruction(self, epoch=None, reconstructed_mass_spec=reconstructed_images,
                                                       original=real_images)
                mz_values_loss_batch_list.append(mz_values_loss)
                intensities_loss_batch_list.append(intensities_loss)

            # store loss in list
            autoencoder_loss_batch_list.append(autoencoder_loss)
            discriminator_loss_batch_list.append(discriminator_loss)
            generator_loss_batch_list.append(generator_loss)

        # calculate the avg loss
        autoencoder_loss_final = np.mean(autoencoder_loss_batch_list)
        discriminator_loss_final = np.mean(discriminator_loss_batch_list)
        generator_loss_final = np.mean(generator_loss_batch_list)

        # print the final losses
        if self.verbose:
            print("Autoencoder Loss: {}".format(autoencoder_loss_final))
            print("Discriminator Loss: {}".format(discriminator_loss_final))
            print("Generator Loss: {}".format(generator_loss_final))
            if self.selected_dataset == "mass_spec":
                print("M/z Loss: {}".format(np.mean(mz_values_loss_batch_list)))
                print("Intensities Loss: {}".format(np.mean(intensities_loss_batch_list)))
            print("#############    FINISHED TRAINING    #############")

        if log_path is not None:
            with open(log_path + '/log.txt', 'a') as log:
                log.write("Epoch: Final\n")
                log.write("Autoencoder Loss: {}\n".format(autoencoder_loss_final))
                log.write("Discriminator Loss: {}\n".format(discriminator_loss_final))
                log.write("Generator Loss: {}\n".format(generator_loss_final))

            if self.selected_dataset == "mass_spec":
                with open(log_path + '/log_mass_spec_reconstruction.txt', 'a') as log:
                    log.write("Epoch: Final\n")
                    log.write("M/Z values Loss: {}\n".format(np.mean(mz_values_loss_batch_list)))
                    log.write("Intensities Loss: {}\n".format(np.mean(intensities_loss_batch_list)))

        # set the final performance
        self.final_performance = {"autoencoder_loss_final": autoencoder_loss_final,
                                  "discriminator_loss_final": discriminator_loss_final,
                                  "generator_loss_final": generator_loss_final,
                                  "summed_loss_final": autoencoder_loss_final + discriminator_loss_final
                                                       + generator_loss_final}

        # create the gif for the learning progress
        create_gif(self)

        # training has stopped
        self.train_status = "stop"

        # close the tensorflow session
        sess.close()
