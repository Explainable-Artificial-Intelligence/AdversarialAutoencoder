"""
    Implementation of a SemiSupervised Adversarial Autoencoder based on the Paper Adversarial Autoencoders
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
from util.DataLoading import get_input_data
from util.Distributions import draw_from_single_gaussian
from util.NeuralNetworkUtils import get_loss_function, get_optimizer, get_layer_names, create_dense_layer, form_results, \
    get_learning_rate_for_optimizer, get_biases_or_weights_for_layer
from util.VisualizationUtils import reshape_tensor_to_rgb_image, reshape_image_array, \
    draw_class_distribution_on_latent_space, visualize_autoencoder_weights_and_biases, \
    create_gif, visualize_cluster_heads, \
    create_epoch_summary_image_unsupervised_clustering


class UnsupervisedClusteringAdversarialAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameter_dictionary):

        self.n_clusters = 16

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

        self.n_classes = parameter_dictionary["n_classes"]
        self.n_labeled = parameter_dictionary["n_labeled"]

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
        self.mass_spec_data_properties = parameter_dictionary["mass_spec_data_properties"]

        """
        params for network topology
        """

        # number of neurons of the hidden layers
        self.n_neurons_of_hidden_layer_x_autoencoder = parameter_dictionary["n_neurons_of_hidden_layer_x_autoencoder"]
        self.n_neurons_of_hidden_layer_x_discriminator_c = \
            parameter_dictionary["n_neurons_of_hidden_layer_x_discriminator_c"]
        self.n_neurons_of_hidden_layer_x_discriminator_g = \
            parameter_dictionary["n_neurons_of_hidden_layer_x_discriminator_g"]

        # dropout for the layers
        self.dropout_encoder = tf.placeholder_with_default([0.0]*len(parameter_dictionary["dropout_encoder"]),
                                                           shape=(len(parameter_dictionary["dropout_encoder"]),))
        self.dropout_decoder = tf.placeholder_with_default([0.0]*len(parameter_dictionary["dropout_decoder"]),
                                                           shape=(len(parameter_dictionary["dropout_decoder"]),))
        self.dropout_discriminator_c = \
            tf.placeholder_with_default([0.0]*len(parameter_dictionary["dropout_discriminator_c"]),
                                        shape=(len(parameter_dictionary["dropout_discriminator_c"]),))

        self.dropout_discriminator_g = \
            tf.placeholder_with_default([0.0]*len(parameter_dictionary["dropout_discriminator_g"]),
                                        shape=(len(parameter_dictionary["dropout_discriminator_g"]),))

        # what batch normalization to use for the different layers (no BN, post-activation, pre-activation)
        self.batch_normalization_encoder = parameter_dictionary["batch_normalization_encoder"]
        self.batch_normalization_decoder = parameter_dictionary["batch_normalization_decoder"]
        self.batch_normalization_discriminator_c = parameter_dictionary["batch_normalization_discriminator_c"]
        self.batch_normalization_discriminator_g = parameter_dictionary["batch_normalization_discriminator_g"]

        # convert "None" to None
        self.batch_normalization_encoder = [x if x is not "None" else None for x in self.batch_normalization_encoder]
        self.batch_normalization_decoder = [x if x is not "None" else None for x in self.batch_normalization_decoder]
        self.batch_normalization_discriminator_c = [x if x is not "None" else None for x in
                                                    self.batch_normalization_discriminator_c]
        self.batch_normalization_discriminator_g = [x if x is not "None" else None for x in
                                                    self.batch_normalization_discriminator_g]

        # how the biases of the different layers should be initialized
        self.bias_initializer_encoder = parameter_dictionary["bias_initializer_encoder"]
        self.bias_initializer_decoder = parameter_dictionary["bias_initializer_decoder"]
        self.bias_initializer_discriminator_c = parameter_dictionary["bias_initializer_discriminator_c"]
        self.bias_initializer_discriminator_g = parameter_dictionary["bias_initializer_discriminator_g"]

        # parameters for the initialization of the different layers, e.g. mean and stddev for the
        # random_normal_initializer
        self.bias_initializer_params_encoder = parameter_dictionary["bias_initializer_params_encoder"]
        self.bias_initializer_params_decoder = parameter_dictionary["bias_initializer_params_decoder"]
        self.bias_initializer_params_discriminator_c = parameter_dictionary["bias_initializer_params_discriminator_c"]
        self.bias_initializer_params_discriminator_g = parameter_dictionary["bias_initializer_params_discriminator_g"]

        # how the weights of the different layers should be initialized
        self.weights_initializer_encoder = parameter_dictionary["weights_initializer_encoder"]
        self.weights_initializer_decoder = parameter_dictionary["weights_initializer_decoder"]
        self.weights_initializer_discriminator_c = parameter_dictionary["weights_initializer_discriminator_c"]
        self.weights_initializer_discriminator_g = parameter_dictionary["weights_initializer_discriminator_g"]

        # parameters for the initialization of the different layers, e.g. mean and stddev for the
        # random_normal_initializer
        self.weights_initializer_params_encoder = parameter_dictionary["weights_initializer_params_encoder"]
        self.weights_initializer_params_decoder = parameter_dictionary["weights_initializer_params_decoder"]
        self.weights_initializer_params_discriminator_c = \
            parameter_dictionary["weights_initializer_params_discriminator_c"]
        self.weights_initializer_params_discriminator_g = \
            parameter_dictionary["weights_initializer_params_discriminator_g"]

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

        if type(parameter_dictionary["activation_function_discriminator_g"]) is list:
            self.activation_function_discriminator_g = parameter_dictionary["activation_function_discriminator_g"]
        else:
            self.activation_function_discriminator_g \
                = [parameter_dictionary["activation_function_discriminator_g"]] * \
                  (len(self.n_neurons_of_hidden_layer_x_discriminator_g) + 1)

        if type(parameter_dictionary["activation_function_discriminator_c"]) is list:
            self.activation_function_discriminator_c = parameter_dictionary["activation_function_discriminator_c"]
        else:
            self.activation_function_discriminator_c \
                = [parameter_dictionary["activation_function_discriminator_c"]] * \
                  (len(self.n_neurons_of_hidden_layer_x_discriminator_c) + 1)

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

        # initial learning rate for the different parts of the network
        self.learning_rate_autoencoder = parameter_dictionary["learning_rate_autoencoder"]
        self.learning_rate_discriminator_gaussian = parameter_dictionary["learning_rate_discriminator_gaussian"]
        self.learning_rate_discriminator_categorical = parameter_dictionary["learning_rate_discriminator_categorical"]
        self.learning_rate_generator = parameter_dictionary["learning_rate_generator"]
        self.learning_rate_supervised_encoder = parameter_dictionary["learning_rate_supervised_encoder"]

        # initial learning rate for the different parts of the network
        self.decaying_learning_rate_name_autoencoder = parameter_dictionary["decaying_learning_rate_name_autoencoder"]
        self.decaying_learning_rate_name_discriminator_gaussian = \
            parameter_dictionary["decaying_learning_rate_name_discriminator_gaussian"]
        self.decaying_learning_rate_name_discriminator_categorical = \
            parameter_dictionary["decaying_learning_rate_name_discriminator_categorical"]
        self.decaying_learning_rate_name_generator = parameter_dictionary["decaying_learning_rate_name_generator"]
        self.decaying_learning_rate_name_supervised_encoder = \
            parameter_dictionary["decaying_learning_rate_name_supervised_encoder"]

        """
        loss functions
        """

        # loss function
        self.loss_function_discriminator_c = parameter_dictionary["loss_function_discriminator_categorical"]
        self.loss_function_discriminator_g = parameter_dictionary["loss_function_discriminator_gaussian"]
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

        # holds a single labeled input image
        self.X_labeled_single_image = tf.placeholder(dtype=tf.float32, shape=[1, self.input_dim],
                                                     name='Single_Labeled_Input')

        # holds a single label
        self.single_label = tf.placeholder(dtype=tf.float32, shape=[1, self.n_classes], name='Single_label')

        # holds the desired output of the autoencoder
        self.X_target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Target')

        # holds the real distribution p(z) used as positive sample for the discriminator
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim],
                                                name='Real_distribution')

        # holds the input samples for the decoder (only for generating the images; NOT used for training)
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim + self.n_clusters],
                                            name='Decoder_input')
        self.decoder_input_multiple = \
            tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim + self.n_clusters],
                           name='Decoder_input_multiple')

        # holds the categorical distribution
        self.categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_clusters],
                                                      name='Categorical_distribution')

        # for proper batch normalization
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        """
        Init the network; generator doesn't need to be initiated, since the generator is the encoder of the autoencoder
        """

        # init autoencoder
        with tf.variable_scope(tf.get_variable_scope()):
            # encoder q(y,z|x) has two parts: latent space part coding the style information q(z|x) and the class label
            # part q(y|x)
            encoder = self.encoder(self.X)
            # q(z|x): encodes the latent representation (style information)
            self.encoder_latent_space = encoder[0]
            # q(y|x): encodes the class labels
            self.encoder_class_label = encoder[1]

            # Concat class label and the encoder output
            decoder_input = tf.concat([self.encoder_class_label, self.encoder_latent_space], 1)
            # decoder part of the autoencoder; takes z and y as input
            decoder_output = self.decoder(decoder_input)

        # init discriminator for the samples drawn from some gaussian distribution (holds style information)
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive gaussian samples drawn from N(z|0,I)
            self.discriminator_gaussian_pos_samples = \
                self.discriminator_gaussian(self.real_distribution)
            # discriminator for the negative gaussian samples q(z) (generated by the generator)
            self.discriminator_gaussian_neg_samples = \
                self.discriminator_gaussian(self.encoder_latent_space, reuse=True)

        # init discriminator for the samples drawn from the categorical distribution (holds class label information)
        with tf.variable_scope(tf.get_variable_scope()):
            # discriminator for the positive categorical samples drawn from Cat(y)
            self.discriminator_categorical_pos_samples = \
                self.discriminator_categorical(self.categorial_distribution)
            # discriminator for the negative categorical samples (= predicted labels y) (generated by the
            # generator)
            self.discriminator_categorical_neg_samples = \
                self.discriminator_categorical(self.encoder_class_label, reuse=True)

        # variable for predicting the class labels from the labeled data (for performance evaluation)
        with tf.variable_scope(tf.get_variable_scope()):
            # predict the labels by passing the data through the encoder part
            _, predicted_labels = \
                self.encoder(self.X_labeled,
                             reuse=True, is_supervised=True)

            # classify a single image
            _, self.predicted_label_single_image = \
                self.encoder(self.X_labeled_single_image, reuse=True,
                             is_supervised=True)

        # variable for "manually" passing values through the decoder (currently not in use -> later when clicking on
        # distribution -> show respective image)
        with tf.variable_scope(tf.get_variable_scope()):
            self.decoder_output_real_dist = self.decoder(self.decoder_input, reuse=True)

            self.decoder_output_multiple = \
                self.decoder(self.decoder_input_multiple, reuse=True)

        """
        Init the loss functions
        """

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_target - decoder_output))

        # Gaussian Discriminator Loss
        discriminator_gaussian_loss_pos_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator_g,
                                         labels=tf.ones_like(self.discriminator_gaussian_pos_samples),
                                         logits=self.discriminator_gaussian_pos_samples))
        discriminator_gaussian_loss_neg_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator_g,
                                         labels=tf.zeros_like(self.discriminator_gaussian_neg_samples),
                                         logits=self.discriminator_gaussian_neg_samples))
        self.discriminator_gaussian_loss = discriminator_gaussian_loss_neg_samples + \
                                           discriminator_gaussian_loss_pos_samples

        # Categorical Discrimminator Loss
        discriminator_categorical_loss_pos_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator_c,
                                         labels=tf.ones_like(self.discriminator_categorical_pos_samples),
                                         logits=self.discriminator_categorical_pos_samples))
        discriminator_categorical_loss_neg_samples = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_discriminator_c,
                                         labels=tf.zeros_like(self.discriminator_categorical_neg_samples),
                                         logits=self.discriminator_categorical_neg_samples))
        self.discriminator_categorical_loss = discriminator_categorical_loss_pos_samples + \
                                              discriminator_categorical_loss_neg_samples

        # Generator loss
        generator_gaussian_loss = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_generator,
                                         labels=tf.ones_like(self.discriminator_gaussian_neg_samples),
                                         logits=self.discriminator_gaussian_neg_samples))
        generator_categorical_loss = tf.reduce_mean(
            get_loss_function(loss_function=self.loss_function_generator,
                                         labels=tf.ones_like(self.discriminator_categorical_neg_samples),
                                         logits=self.discriminator_categorical_neg_samples))
        self.generator_loss = generator_gaussian_loss + generator_categorical_loss

        """
        Init the optimizers
        """

        optimizer_autoencoder = parameter_dictionary["optimizer_autoencoder"]
        optimizer_discriminator_gaussian = parameter_dictionary["optimizer_discriminator_gaussian"]
        optimizer_discriminator_categorical = parameter_dictionary["optimizer_discriminator_categorical"]
        optimizer_generator = parameter_dictionary["optimizer_generator"]

        # get the discriminator and encoder variables
        all_variables = tf.trainable_variables()
        discriminator_gaussian_vars = [var for var in all_variables if 'discriminator_gaussian' in var.name]
        discriminator_categorical_vars = [var for var in all_variables if 'discriminator_categorical' in var.name]
        encoder_vars = [var for var in all_variables if 'encoder_' in var.name]

        # Optimizers
        self.autoencoder_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_autoencoder, "autoencoder", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_autoencoder)
        self.autoencoder_trainer = self.autoencoder_optimizer.minimize(self.autoencoder_loss)

        self.discriminator_gaussian_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_discriminator_gaussian, "discriminator_gaussian",
                          global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator_gaussian)
        self.discriminator_gaussian_trainer = \
            self.discriminator_gaussian_optimizer.minimize(self.discriminator_gaussian_loss,
                                                           var_list=discriminator_gaussian_vars)

        self.discriminator_categorical_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_discriminator_categorical, "discriminator_categorical",
                          global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_discriminator_categorical)
        self.discriminator_categorical_trainer = \
            self.discriminator_categorical_optimizer.minimize(self.discriminator_categorical_loss,
                                                              var_list=discriminator_categorical_vars)

        self.generator_optimizer = \
            get_optimizer(self.parameter_dictionary, optimizer_generator, "generator", global_step=self.global_step,
                          decaying_learning_rate_name=self.decaying_learning_rate_name_generator)
        self.generator_trainer = \
            self.generator_optimizer.minimize(self.generator_loss, var_list=encoder_vars)

        """
        Create the tensorboard summary and the tf.saver and tf.session vars
        """
        self.tensorboard_summary = \
            self.create_tensorboard_summary(decoder_output=decoder_output, encoder_output=self.encoder_latent_space,
                                            autoencoder_loss=self.autoencoder_loss,
                                            discriminator_gaussian_loss=self.discriminator_gaussian_loss,
                                            discriminator_categorical_loss=self.discriminator_categorical_loss,
                                            generator_loss=self.generator_loss,
                                            real_distribution=self.real_distribution,
                                            encoder_output_label=self.encoder_class_label,
                                            categorical_distribution=self.categorial_distribution,
                                            decoder_output_multiple=self.decoder_output_multiple)

        # for saving the model
        self.saver = tf.train.Saver()

        # holds the tensorflow session
        self.session = tf.Session()

        """
        Variable for the "manual" summary 
        """

        self.final_performance = None
        self.performance_over_time = {"autoencoder_losses": [], "discriminator_gaussian_losses": [],
                                      "discriminator_categorical_losses": [], "generator_losses": [],
                                      "supervised_encoder_loss": [], "list_of_epochs": []}
        self.learning_rates = {"autoencoder_lr": [], "discriminator_g_lr": [], "discriminator_c_lr": [],
                               "generator_lr": [], "supervised_encoder_lr": [], "list_of_epochs": []}

        self.epoch_summary_vars = {"real_dist": [], "latent_representation": [],
                                   "batch_x": [],
                                   "reconstructed_images": [], "epoch": None,
                                   "real_cat_dist": [], "encoder_cat_dist": [],
                                   "batch_labels": [], "discriminator_gaussian_neg": [],
                                   "discriminator_gaussian_pos": [], "discriminator_cat_neg": [],
                                   "discriminator_cat_pos": []}

        # only for tuning; if set to true, the previous tuning results (losses and learning rates) are included in the
        # minibatch summary plots
        self.include_tuning_performance = False

        # holds the generated random points used for the image grid
        self.random_points_for_image_grid = None

        # holds the names for all layers
        self.all_layer_names = get_layer_names(self)

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

    def get_performance(self):
        return self.final_performance

    def get_result_folder_name(self):
        return self.result_folder_name

    def encoder(self, X, reuse=False, is_supervised=False):
        """
        Encoder of the autoencoder.
        :param X: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :param is_supervised: True -> returns output for encoder part which encodes the labels,
                              False -> returns output for encoder part which encodes the latent representation
                              (=style information).
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
                latent_variable_z = \
                    create_dense_layer(X, self.input_dim, self.z_dim, 'encoder_output',
                                                  activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[0],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[0],
                                                  bias_initializer=self.bias_initializer_encoder[0],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[0],
                                                  batch_normalization=self.batch_normalization_encoder[0],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)

                categorical_encoder_label = \
                    create_dense_layer(latent_variable_z, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                  self.n_classes, 'encoder_label', activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[1],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[1],
                                                  bias_initializer=self.bias_initializer_encoder[1],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[1],
                                                  batch_normalization=self.batch_normalization_encoder[1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)

                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

            # there is only one hidden layer
            elif n_hidden_layers == 1:
                dense_layer_1 = \
                    create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                  'encoder_dense_layer_1',
                                                  activation_function=self.activation_function_encoder[0],
                                                  weight_initializer=self.weights_initializer_encoder[0],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[0],
                                                  bias_initializer=self.bias_initializer_encoder[0],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[0],
                                                  batch_normalization=self.batch_normalization_encoder[0],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)
                latent_variable_z = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                  self.z_dim, 'encoder_output', activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[0],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[0],
                                                  bias_initializer=self.bias_initializer_encoder[0],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[0],
                                                  batch_normalization=self.batch_normalization_encoder[0],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)

                categorical_encoder_label = \
                    create_dense_layer(latent_variable_z, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                  self.n_classes, 'encoder_label', activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[-1],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[-1],
                                                  bias_initializer=self.bias_initializer_encoder[-1],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[-1],
                                                  batch_normalization=self.batch_normalization_encoder[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)
                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.input_dim, self.n_neurons_of_hidden_layer_x_autoencoder[0],
                                                  'encoder_dense_layer_1',
                                                  activation_function=self.activation_function_encoder[0],
                                                  weight_initializer=self.weights_initializer_encoder[0],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[0],
                                                  bias_initializer=self.bias_initializer_encoder[0],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[0],
                                                  batch_normalization=self.batch_normalization_encoder[0],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)
                for i in range(1, n_hidden_layers):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i,
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i - 1],
                                                      self.n_neurons_of_hidden_layer_x_autoencoder[i],
                                                      'encoder_dense_layer_' + str(i + 1),
                                                      activation_function=self.activation_function_encoder[i],
                                                      weight_initializer=self.weights_initializer_encoder[i],
                                                      weight_initializer_params=self.weights_initializer_params_encoder[i],
                                                      bias_initializer=self.bias_initializer_encoder[i],
                                                      bias_initializer_params=self.bias_initializer_params_encoder[i],
                                                      batch_normalization=self.batch_normalization_encoder[i],
                                                      drop_out_rate_input_layer=0.0,
                                                      drop_out_rate_output_layer=0.0,
                                                      is_training=self.is_training)
                latent_variable_z = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                  self.z_dim, 'encoder_output', activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[-1],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[-1],
                                                  bias_initializer=self.bias_initializer_encoder[-1],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[-1],
                                                  batch_normalization=self.batch_normalization_encoder[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)

                # label prediction of the encoder
                categorical_encoder_label = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_autoencoder[-1],
                                                  self.n_clusters, 'encoder_label', activation_function="linear",
                                                  weight_initializer=self.weights_initializer_encoder[-1],
                                                  weight_initializer_params=self.weights_initializer_params_encoder[-1],
                                                  bias_initializer=self.bias_initializer_encoder[-1],
                                                  bias_initializer_params=self.bias_initializer_params_encoder[-1],
                                                  batch_normalization=self.batch_normalization_encoder[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=0.0,
                                                  is_training=self.is_training)
                if not is_supervised:
                    # normalize the encoder label tensor (= assign probabilities to it)
                    # TODO: parameter for softmax
                    softmax_label = tf.nn.softmax(logits=categorical_encoder_label, name='e_softmax_label')
                else:
                    softmax_label = categorical_encoder_label
                return latent_variable_z, softmax_label

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
                    create_dense_layer(X, self.z_dim + self.n_clusters, self.input_dim, 'x_reconstructed',
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
                    create_dense_layer(X, self.z_dim + self.n_clusters,
                                                  self.n_neurons_of_hidden_layer_x_autoencoder[0],
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
                                                  self.input_dim,
                                                  'x_reconstructed',
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
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.z_dim + self.n_clusters,
                                                  self.n_neurons_of_hidden_layer_x_autoencoder[-1],
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
                                                      drop_out_rate_output_layer=self.dropout_decoder[i+1],
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

    def discriminator_gaussian(self, X, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given gaussian prior distribution.
        :param X: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """

        # number of hidden layers
        n__hidden_layers = len(self.n_neurons_of_hidden_layer_x_discriminator_g)

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Discriminator'
        with tf.name_scope('Discriminator_Gaussian'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = \
                    create_dense_layer(X, self.z_dim, 1, 'discriminator_gaussian_output',
                                                  weight_initializer=self.weights_initializer_discriminator_g[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[0],
                                                  bias_initializer=self.bias_initializer_discriminator_g[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[0],
                                                  activation_function=self.activation_function_discriminator_g[0],
                                                  batch_normalization=self.batch_normalization_discriminator_g[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_g[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[1],
                                                  is_training=self.is_training)
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator_g[0],
                                                  'discriminator_gaussian_dense_layer_1',
                                                  weight_initializer=self.weights_initializer_discriminator_g[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[0],
                                                  bias_initializer=self.bias_initializer_discriminator_g[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[0],
                                                  activation_function=self.activation_function_discriminator_g[0],
                                                  batch_normalization=self.batch_normalization_discriminator_g[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_g[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[1],
                                                  is_training=self.is_training)
                discriminator_output = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator_g[-1], 1,
                                                  'discriminator_gaussian_output',
                                                  weight_initializer=self.weights_initializer_discriminator_g[-1],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[-1],
                                                  bias_initializer=self.bias_initializer_discriminator_g[-1],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[-1],
                                                  activation_function=self.activation_function_discriminator_g[-1],
                                                  batch_normalization=self.batch_normalization_discriminator_g[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[-1],
                                                  is_training=self.is_training)
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.z_dim, self.n_neurons_of_hidden_layer_x_discriminator_g[0],
                                                  'discriminator_gaussian_dense_layer_1',
                                                  weight_initializer=self.weights_initializer_discriminator_g[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[0],
                                                  bias_initializer=self.bias_initializer_discriminator_g[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[0],
                                                  activation_function=self.activation_function_discriminator_g[0],
                                                  batch_normalization=self.batch_normalization_discriminator_g[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_g[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[1],
                                                  is_training=self.is_training)
                for i in range(1, n__hidden_layers):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i,
                                                      self.n_neurons_of_hidden_layer_x_discriminator_g[i - 1],
                                                      self.n_neurons_of_hidden_layer_x_discriminator_g[i],
                                                      'discriminator_gaussian_dense_layer_' + str(i + 1),
                                                  weight_initializer=self.weights_initializer_discriminator_g[i],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[i],
                                                  bias_initializer=self.bias_initializer_discriminator_g[i],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[i],
                                                  activation_function=self.activation_function_discriminator_g[i],
                                                  batch_normalization=self.batch_normalization_discriminator_g[i],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[i+1],
                                                  is_training=self.is_training)
                discriminator_output = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator_g[-1],
                                                  1, 'discriminator_gaussian_output',
                                                  weight_initializer=self.weights_initializer_discriminator_g[-1],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_g[-1],
                                                  bias_initializer=self.bias_initializer_discriminator_g[-1],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_g[-1],
                                                  activation_function=self.activation_function_discriminator_g[-1],
                                                  batch_normalization=self.batch_normalization_discriminator_g[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_g[-1],
                                                  is_training=self.is_training)
                return discriminator_output

    def discriminator_categorical(self, X, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given categorical prior distribution.
        :param X: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """

        # number of hidden layers
        n__hidden_layers = len(self.n_neurons_of_hidden_layer_x_discriminator_c)

        # allows tensorflow to reuse the variables defined in the current variable scope
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # create the variables in the variable scope 'Discriminator'
        with tf.name_scope('Discriminator_Categorical'):
            # there is no hidden layer
            if n__hidden_layers == 0:
                discriminator_output = \
                    create_dense_layer(X, self.n_clusters, 1, 'discriminator_categorical_output',
                                                  weight_initializer=self.weights_initializer_discriminator_c[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[0],
                                                  bias_initializer=self.bias_initializer_discriminator_c[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[0],
                                                  activation_function=self.activation_function_discriminator_c[0],
                                                  batch_normalization=self.batch_normalization_discriminator_c[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_c[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[1],
                                                  is_training=self.is_training)
                return discriminator_output
            # there is only one hidden layer
            elif n__hidden_layers == 1:
                dense_layer_1 = \
                    create_dense_layer(X, self.n_clusters,
                                                  self.n_neurons_of_hidden_layer_x_discriminator_c[0],
                                                  'discriminator_categorical_dense_layer_1',
                                                  weight_initializer=self.weights_initializer_discriminator_c[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[0],
                                                  bias_initializer=self.bias_initializer_discriminator_c[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[0],
                                                  activation_function=self.activation_function_discriminator_c[0],
                                                  batch_normalization=self.batch_normalization_discriminator_c[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_c[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[1],
                                                  is_training=self.is_training)
                discriminator_output = \
                    create_dense_layer(dense_layer_1, self.n_neurons_of_hidden_layer_x_discriminator_c[-1], 1,
                                                  'discriminator_categorical_output',
                                                  weight_initializer=self.weights_initializer_discriminator_c[-1],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[-1],
                                                  bias_initializer=self.bias_initializer_discriminator_c[-1],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[-1],
                                                  activation_function=self.activation_function_discriminator_c[-1],
                                                  batch_normalization=self.batch_normalization_discriminator_c[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[-1],
                                                  is_training=self.is_training)
                return discriminator_output
            # there is an arbitrary number of hidden layers
            else:
                dense_layer_i = \
                    create_dense_layer(X, self.n_clusters,
                                                  self.n_neurons_of_hidden_layer_x_discriminator_c[0],
                                                  'discriminator_categorical_dense_layer_1',
                                                  weight_initializer=self.weights_initializer_discriminator_c[0],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[0],
                                                  bias_initializer=self.bias_initializer_discriminator_c[0],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[0],
                                                  activation_function=self.activation_function_discriminator_c[0],
                                                  batch_normalization=self.batch_normalization_discriminator_c[0],
                                                  drop_out_rate_input_layer=self.dropout_discriminator_c[0],
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[1],
                                                  is_training=self.is_training)
                for i in range(1, n__hidden_layers):
                    dense_layer_i = \
                        create_dense_layer(dense_layer_i,
                                                      self.n_neurons_of_hidden_layer_x_discriminator_c[i - 1],
                                                      self.n_neurons_of_hidden_layer_x_discriminator_c[i],
                                                      'discriminator_categorical_dense_layer_' + str(i + 1),
                                                  weight_initializer=self.weights_initializer_discriminator_c[i],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[i],
                                                  bias_initializer=self.bias_initializer_discriminator_c[i],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[i],
                                                  activation_function=self.activation_function_discriminator_c[i],
                                                  batch_normalization=self.batch_normalization_discriminator_c[i],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[i+1],
                                                  is_training=self.is_training)
                discriminator_output = \
                    create_dense_layer(dense_layer_i, self.n_neurons_of_hidden_layer_x_discriminator_c[-1],
                                                  1, 'discriminator_categorical_output',
                                                  weight_initializer=self.weights_initializer_discriminator_c[-1],
                                                  weight_initializer_params=self.weights_initializer_params_discriminator_c[-1],
                                                  bias_initializer=self.bias_initializer_discriminator_c[-1],
                                                  bias_initializer_params=self.bias_initializer_params_discriminator_c[-1],
                                                  activation_function=self.activation_function_discriminator_c[-1],
                                                  batch_normalization=self.batch_normalization_discriminator_c[-1],
                                                  drop_out_rate_input_layer=0.0,
                                                  drop_out_rate_output_layer=self.dropout_discriminator_c[-1],
                                                  is_training=self.is_training)
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
                                   discriminator_categorical_loss, generator_loss,
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
            input_images = reshape_tensor_to_rgb_image(self.X, self.input_dim_x, self.input_dim_y)
            generated_images = reshape_tensor_to_rgb_image(decoder_output, self.input_dim_x, self.input_dim_y)
            generated_images_z_dist = reshape_tensor_to_rgb_image(decoder_output_multiple, self.input_dim_x,
                                                                                                        self.input_dim_y)
        else:
            input_images = tf.reshape(self.X, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images = tf.reshape(decoder_output, [-1, self.input_dim_x, self.input_dim_y, 1])
            generated_images_z_dist = tf.reshape(decoder_output_multiple, [-1, self.input_dim_x, self.input_dim_y, 1])

        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Discriminator Gaussian Loss', tensor=discriminator_gaussian_loss)
        tf.summary.scalar(name='Discriminator Categorical Loss', tensor=discriminator_categorical_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.histogram(name='Encoder Gaussian Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Gaussian Distribution', values=real_distribution)
        tf.summary.histogram(name='Encoder Categorical Distribution', values=encoder_output_label)
        tf.summary.histogram(name='Real Categorical Distribution', values=categorical_distribution)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=50)
        tf.summary.image(name='Generated Images from input images', tensor=generated_images, max_outputs=50)
        tf.summary.image(name='Generated Images z-dist', tensor=generated_images_z_dist, max_outputs=50)

        summary_op = tf.summary.merge_all()
        return summary_op

    def classify_single_image(self, sess, image_to_classify):
        """
        classifies a single image and returns the integer label of it
        :param sess: tensorflow session
        :param image_to_classify: np array of the image we want to classify
        :return: integer label of the predicted image
        """

        image_to_classify = np.reshape(image_to_classify, (1, self.input_dim))

        predicted_label = sess.run(self.predicted_label_single_image,
                                   feed_dict={self.X_labeled_single_image: image_to_classify, self.is_training: False})

        # convert network output to single integer label
        predicted_label = np.argmax(predicted_label, 1)[0]

        return predicted_label

    def generate_image_grid(self, sess, op, epoch, left_cell=None, save_image_grid=True):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :param epoch: current epoch of the training; image grid is saved as <epoch>.png
        :param left_cell: left cell of the grid spec with two adjacent horizontal cells holding the image grid
        and the class distribution on the latent space; if left_cell is None, then only the image grid is supposed
        to be plotted and not the "combinated" image (image grid + class distr. on latent space).
        :param save_image_grid: whether to save the image grid
        :return: None, displays a matplotlib window with all the merged images.
        """
        nx, ny = self.n_clusters, self.n_clusters
        self.random_points_for_image_grid = np.random.randn(self.n_clusters, self.z_dim) * 5.

        cluster_labels = np.identity(self.n_clusters)

        # create the image grid
        if left_cell:
            gs = gridspec.GridSpecFromSubplotSpec(nx, ny, left_cell)
        else:
            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        # unicode equivalent for the used matplotlib markers
        # TODO: more than 10
        unicode_markers = [u"\u25b2", u"\u25bc", u"\u25b6", u"\u25c0", u"\u25a0", u"\u2605", "-", "+", "x", "|",
                           "a", "b", "c", "d", "e", "f"]

        list_of_images_for_swagger = []

        i = 0
        for cluster_label_one_hot in cluster_labels:
            for j, r in enumerate(self.random_points_for_image_grid):
                r, cluster_label_one_hot = np.reshape(r, (1, self.z_dim)), np.reshape(cluster_label_one_hot,
                                                                                    (1, self.n_clusters))
                dec_input = np.concatenate((cluster_label_one_hot, r), 1)
                x = sess.run(op, feed_dict={self.decoder_input: dec_input, self.is_training: False})
                ax = plt.subplot(gs[i])
                i += 1

                # reshape the images according to the color scale
                img = reshape_image_array(self, x, is_array_of_arrays=True)
                list_of_images_for_swagger.append(img)

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
                    class_label = int(i / self.n_clusters)
                    ax.set_ylabel(class_label, fontsize=9)

                # create the label x for the x axis
                if ax.is_last_row():
                    ax.set_xlabel(unicode_markers[j], fontsize=9)

        if not left_cell and save_image_grid:
            plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')

        return list_of_images_for_swagger

    def generate_image_from_single_point_and_class_label(self, sess, params):
        """
        generates a image based on the point on the latent space and the class label
        :param sess: tensorflow session
        :param params: tuple holding the array of coordinates of the point in latent space and the class label as one
        hot vector
        :return: np array representing the generated image
        """

        # get the point and the class label
        single_point = params[0]
        class_label_one_hot = params[1]

        # reshape the point and the class label
        single_point = np.reshape(single_point, (1, self.z_dim))
        class_label_one_hot = np.reshape(class_label_one_hot, (1, self.n_classes))

        # create the decoder input
        dec_input = np.concatenate((class_label_one_hot, single_point), 1)

        # run the point through the decoder to generate the image
        generated_image = sess.run(self.decoder_output_real_dist, feed_dict={self.decoder_input: dec_input,
                                                                             self.is_training: False})

        # reshape the image array and display it
        generated_image = np.array(generated_image).reshape(self.input_dim)
        img = reshape_image_array(self, generated_image, is_array_of_arrays=True)

        # show the image
        if self.color_scale == "gray_scale":
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.savefig(self.results_path + self.result_folder_name + '/Tensorboard/' + "test_algorithm" + '.png')

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

        # Get the data from the storage class, we have data stored
        if Storage.get_all_input_data():
            data = Storage.get_all_input_data()
        else:
            data = get_input_data(self.selected_dataset, color_scale=self.color_scale, data_normalized=False,
                                  mass_spec_data_properties=self.mass_spec_data_properties)

        autoencoder_loss_final = 0
        discriminator_loss_g_final = 0
        discriminator_loss_c_final = 0
        generator_loss_final = 0
        supervised_encoder_loss_final = 0
        epochs_completed = 0

        autoencoder_epoch_losses, discriminator_gaussian_epoch_losses, discriminator_categorical_epoch_losses, \
        generator_epoch_losses = [], [], [], []

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

                # write the parameter dictionary to some file
                json_dictionary = json.dumps(self.parameter_dictionary)
                with open(log_path + '/params.txt', 'a') as file:
                    file.write(json_dictionary)

                batch_x_labeled, batch_labels_labeled = data.test.next_batch(self.n_labeled)

                # we want n_epochs iterations
                for epoch in range(self.n_epochs):

                    if self.train_status == "stop":
                        # end the training
                        break

                    # calculate the number of batches based on the batch_size and the size of the train set
                    n_batches = int(self.n_labeled / self.batch_size)

                    if self.verbose:
                        print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

                    # iterate over the batches
                    for b in range(1, n_batches + 1):

                        self.process_requested_swagger_operations(sess)

                        # draw a sample from p(z) and use it as real distribution for the discriminator
                        z_real_dist = \
                            draw_from_single_gaussian(mean=0.0, std_dev=1.0, shape=(self.batch_size, self.z_dim)) * 5
                        # z_real_dist = draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(self.batch_size,
                        #                                                                          self.z_dim))

                        # create some onehot vectors and use it as input for the categorical discriminator
                        real_cat_dist = np.random.randint(low=0, high=10, size=self.batch_size)
                        real_cat_dist = np.eye(self.n_clusters)[real_cat_dist]

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
                        # train the autoencoder by minimizing the reconstruction error between X_unlabeled and X_target_unlabeled
                        sess.run(self.autoencoder_trainer,
                                 feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                            self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                            self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                            self.dropout_discriminator_c:
                                                self.parameter_dictionary["dropout_discriminator_c"],
                                            self.dropout_discriminator_g:
                                                self.parameter_dictionary["dropout_discriminator_g"],
                                            self.is_training: True})

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
                                            self.real_distribution: z_real_dist,
                                            self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                            self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                            self.dropout_discriminator_c:
                                                self.parameter_dictionary["dropout_discriminator_c"],
                                            self.dropout_discriminator_g:
                                                self.parameter_dictionary["dropout_discriminator_g"],
                                            self.is_training: True})
                        sess.run(self.discriminator_categorical_trainer,
                                 feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                            self.categorial_distribution: real_cat_dist,
                                            self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                            self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                            self.dropout_discriminator_c:
                                                self.parameter_dictionary["dropout_discriminator_c"],
                                            self.dropout_discriminator_g:
                                                self.parameter_dictionary["dropout_discriminator_g"],
                                            self.is_training: True})
                        # train the generator to fool the discriminator with its generated samples.
                        sess.run(self.generator_trainer,
                                 feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                            self.dropout_encoder: self.parameter_dictionary["dropout_encoder"],
                                            self.dropout_decoder: self.parameter_dictionary["dropout_decoder"],
                                            self.dropout_discriminator_c:
                                                self.parameter_dictionary["dropout_discriminator_c"],
                                            self.dropout_discriminator_g:
                                                self.parameter_dictionary["dropout_discriminator_g"],
                                            self.is_training: True})

                        # every 5 epochs: write a summary for every 10th minibatch
                        if epoch % self.summary_image_frequency == 0 and b % 50 == 0:

                            # prepare the decoder inputs
                            n_images_per_class = int(np.ceil(self.batch_size / self.n_clusters))
                            cluster_labels_one_hot = np.identity(self.n_clusters)
                            dec_inputs = []
                            for k in range(n_images_per_class):
                                for cluster_label in cluster_labels_one_hot:
                                    random_inputs = np.random.randn(1, self.z_dim) * 5.
                                    random_inputs = np.reshape(random_inputs, (1, self.z_dim))
                                    cluster_label = np.reshape(cluster_label, (1, self.n_clusters))
                                    dec_inputs.append(np.concatenate((random_inputs, cluster_label), 1))
                            dec_inputs = np.array(dec_inputs)
                            # select batch size entries from the array, so reshape works properly in case batch_size is
                            # not divisible by the number of clusters without remainder
                            random_indices = np.random.choice(dec_inputs.shape[0], self.batch_size, replace=False)
                            random_samples = dec_inputs[random_indices]
                            dec_inputs = random_samples.reshape(self.batch_size, self.z_dim + self.n_clusters)

                            # get the network output for the summary images
                            autoencoder_loss, discriminator_gaussian_loss, discriminator_categorical_loss, \
                            generator_loss, summary, real_dist, \
                            latent_representation, encoder_cat_dist, discriminator_gaussian_pos, \
                            discriminator_gaussian_neg, discriminator_cat_pos, discriminator_cat_neg = sess.run(
                                [self.autoencoder_loss, self.discriminator_gaussian_loss,
                                 self.discriminator_categorical_loss, self.generator_loss,
                                 self.tensorboard_summary, self.real_distribution, self.encoder_latent_space,
                                 self.encoder_class_label, self.discriminator_gaussian_pos_samples,
                                 self.discriminator_gaussian_neg_samples, self.discriminator_categorical_pos_samples,
                                 self.discriminator_categorical_neg_samples],
                                feed_dict={self.X: batch_X_unlabeled, self.X_target: batch_X_unlabeled,
                                           self.real_distribution: z_real_dist, self.y: mini_batch_labels,
                                           self.X_labeled: mini_batch_X_labeled,
                                           self.categorial_distribution: real_cat_dist,
                                           self.decoder_input_multiple: dec_inputs, self.is_training: False})
                            if self.write_tensorboard:
                                writer.add_summary(summary, global_step=step)

                            # prepare the decoder input for a single input image
                            r = np.reshape(latent_representation[0, :], (1, self.z_dim))
                            int_label = np.argmax(batch_X_unlabeled_labels[0, :], axis=0)
                            class_label_one_hot = np.reshape(np.identity(self.n_clusters)[int_label, :], (1, self.n_clusters))
                            # class_label_one_hot = np.reshape(batch_X_unlabeled_labels[0, :], (1, self.n_classes))
                            dec_input = np.concatenate((class_label_one_hot, r), 1)

                            # reconstruct the image
                            reconstructed_images = sess.run(self.decoder_output_real_dist,
                                                           feed_dict={self.decoder_input: dec_input,
                                                                      self.is_training: False})

                            #  digits generated by fixing the style variable to zero and setting the label variable to
                            # one of the 16 one-hot vectors.
                            cluster_head_one_hot_vectors = np.identity(self.n_clusters)
                            cluster_heads = []
                            for cluster_head_one_hot_vector in cluster_head_one_hot_vectors:
                                style_variable = np.reshape(np.array([0]*self.z_dim), (1, self.z_dim))
                                cluster_head_one_hot_vector = cluster_head_one_hot_vector.reshape(1, self.n_clusters)
                                dec_input = np.concatenate((cluster_head_one_hot_vector, style_variable), 1)

                                cluster_head = sess.run(self.decoder_output_real_dist,
                                                        feed_dict={self.decoder_input: dec_input,
                                                                   self.is_training: False})
                                cluster_heads.append(cluster_head)
                            visualize_cluster_heads(self, cluster_heads, epoch, b)

                            # update the lists holding the losses for each epoch
                            autoencoder_epoch_losses.append(autoencoder_loss)
                            discriminator_gaussian_epoch_losses.append(discriminator_gaussian_loss)
                            discriminator_categorical_epoch_losses.append(discriminator_categorical_loss)
                            generator_epoch_losses.append(generator_loss)

                            # update the dictionary holding the learning rates
                            self.learning_rates["autoencoder_lr"].append(
                                get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess))
                            self.learning_rates["discriminator_g_lr"].append(
                                get_learning_rate_for_optimizer(self.discriminator_gaussian_optimizer, sess))
                            self.learning_rates["discriminator_c_lr"].append(
                                get_learning_rate_for_optimizer(self.discriminator_categorical_optimizer, sess))
                            self.learning_rates["generator_lr"].append(
                                get_learning_rate_for_optimizer(self.generator_optimizer, sess))
                            self.learning_rates["list_of_epochs"].append(epoch + (b / n_batches))

                            # update the lists holding the latent representation + labels for the current minibatch
                            latent_representations_current_epoch.extend(latent_representation)
                            labels_current_epoch.extend(batch_X_unlabeled_labels)

                            # updates vars for the swagger server
                            self.epoch_summary_vars["real_dist"].extend(real_dist)
                            self.epoch_summary_vars["latent_representation"].extend(latent_representation)
                            self.epoch_summary_vars["batch_x"].extend(batch_X_unlabeled)
                            self.epoch_summary_vars["reconstructed_images"].extend(reconstructed_images)
                            self.epoch_summary_vars["epoch"] = epoch
                            self.epoch_summary_vars["real_cat_dist"].extend(real_cat_dist)
                            self.epoch_summary_vars["encoder_cat_dist"].extend(encoder_cat_dist)
                            self.epoch_summary_vars["batch_labels"].extend(batch_X_unlabeled_labels)
                            self.epoch_summary_vars["discriminator_gaussian_neg"].extend(discriminator_gaussian_neg)
                            self.epoch_summary_vars["discriminator_gaussian_pos"].extend(discriminator_gaussian_pos)
                            self.epoch_summary_vars["discriminator_cat_neg"].extend(discriminator_cat_neg)
                            self.epoch_summary_vars["discriminator_cat_pos"].extend(discriminator_cat_pos)

                            # set the latest loss as final loss
                            autoencoder_loss_final = autoencoder_loss
                            discriminator_loss_g_final = discriminator_gaussian_loss
                            discriminator_loss_c_final = discriminator_categorical_loss
                            generator_loss_final = generator_loss

                            if self.verbose:
                                print("Epoch: {}, iteration: {}".format(epoch, b))
                                print("Autoencoder Loss: {}".format(autoencoder_loss))
                                print("Discriminator Gauss Loss: {}".format(discriminator_gaussian_loss))
                                print("Discriminator Categorical Loss: {}".format(discriminator_categorical_loss))
                                print("Generator Loss: {}".format(generator_loss))
                                print('Learning rate autoencoder: {}'.format(
                                    get_learning_rate_for_optimizer(self.autoencoder_optimizer, sess)))
                                print('Learning rate categorical discriminator: {}'.format(
                                    get_learning_rate_for_optimizer(self.discriminator_categorical_optimizer,
                                                                               sess)))
                                print('Learning rate gaussian discriminator: {}'.format(
                                    get_learning_rate_for_optimizer(self.discriminator_gaussian_optimizer,
                                                                               sess)))
                                print('Learning rate generator: {}'.format(
                                    get_learning_rate_for_optimizer(self.generator_optimizer, sess)))

                        step += 1

                    # increment the global step:
                    sess.run(self.increment_global_step_op)
                    epochs_completed += 1

                    # every x epochs..
                    if epoch % self.summary_image_frequency == 0:

                        # update the dictionary holding the losses
                        self.performance_over_time["autoencoder_losses"].append(np.mean(autoencoder_epoch_losses))
                        self.performance_over_time["discriminator_categorical_losses"].append(np.mean(discriminator_gaussian_epoch_losses))
                        self.performance_over_time["discriminator_gaussian_losses"].append(np.mean(discriminator_categorical_epoch_losses))
                        self.performance_over_time["generator_losses"].append(np.mean(generator_epoch_losses))
                        self.performance_over_time["list_of_epochs"].append(epoch)

                        autoencoder_epoch_losses, discriminator_gaussian_epoch_losses, discriminator_categorical_epoch_losses, \
                        generator_epoch_losses = [], [], [], []

                        # create the summary image for the current minibatch
                        create_epoch_summary_image_unsupervised_clustering(self,
                                                                           epoch, include_tuning_performance=
                                                                           self.include_tuning_performance)

                        # increase figure size
                        plt.rcParams["figure.figsize"] = (6.4*2, 4.8)
                        outer_grid = gridspec.GridSpec(1, 2)
                        left_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid

                        self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch=epoch,
                                                 left_cell=left_cell)

                        if len(labels_current_epoch) > 0:
                            result_path = self.results_path + self.result_folder_name + '/Tensorboard/'
                            draw_class_distribution_on_latent_space(latent_representations_current_epoch,
                                                                               labels_current_epoch, result_path, epoch,
                                                                               self.random_points_for_image_grid,
                                                                               combined_plot=True)
                            # reset random points
                            self.random_points_for_image_grid = None

                        """
                        Weights + biases visualization
                        """
                        visualize_autoencoder_weights_and_biases(self, epoch=epoch)

                    self.epoch_summary_vars = {"real_dist": [], "latent_representation": [],
                                               "batch_x": [],
                                               "reconstructed_images": [], "epoch": None,
                                               "real_cat_dist": [], "encoder_cat_dist": [],
                                               "batch_labels": [], "discriminator_gaussian_neg": [],
                                               "discriminator_gaussian_pos": [], "discriminator_cat_neg": [],
                                               "discriminator_cat_pos": []}

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

                self.process_requested_swagger_operations(sess)

                self.generate_image_grid(sess, op=self.decoder_output_real_dist, epoch="last")

            if epochs_completed > 0:
                # end the training
                self.end_training(autoencoder_loss_final, discriminator_loss_g_final, discriminator_loss_c_final,
                                  generator_loss_final, supervised_encoder_loss_final, saved_model_path, self.saver,
                                  sess, step)

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
                                                          left_cell=None, save_image_grid=False)
                        self.set_requested_operations_by_swagger_results(result)
                    elif function_name == "generate_image_from_single_point_and_single_label":
                        result = self.generate_image_from_single_point_and_class_label(sess, function_params)
                        self.set_requested_operations_by_swagger_results(result)
                    elif function_name == "classify_single_image":
                        result = self.classify_single_image(sess, function_params)
                        self.set_requested_operations_by_swagger_results(result)
                    elif function_name == "get_biases_or_weights_for_layer":
                        result = get_biases_or_weights_for_layer(self, function_params)
                        self.set_requested_operations_by_swagger_results(result)

                    plt.close('all')

            # reset the list
            self.requested_operations_by_swagger = []

    def end_training(self, autoencoder_loss_final, discriminator_loss_g_final, discriminator_loss_c_final,
                     generator_loss_final, supervised_encoder_loss_final, saved_model_path, saver, sess,
                     step):
        """
        ends the training by saving the model if a model path is provided, saving the final losses and closing the
        tf session
        :param autoencoder_loss_final: final loss of the autoencoder
        :param discriminator_loss_g_final: final loss of the gaussian discriminator
        :param discriminator_loss_c_final: final loss of the categorical discriminator
        :param generator_loss_final: final loss of the generator
        :param supervised_encoder_loss_final: final loss of the supervised encoder
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
            print("Discriminator Gaussian Loss: {}".format(discriminator_loss_g_final))
            print("Discriminator Categorical Loss: {}".format(discriminator_loss_c_final))
            print("Generator Loss: {}".format(generator_loss_final))
            print("Supervised Loss: {}\n".format(supervised_encoder_loss_final))
            print("#############    FINISHED TRAINING   #############")

        # set the final performance
        self.final_performance = {"autoencoder_loss_final": autoencoder_loss_final,
                                  "discriminator_loss_final": discriminator_loss_g_final,
                                  "generator_loss_final": generator_loss_final,
                                  "summed_loss_final": autoencoder_loss_final + discriminator_loss_g_final +
                                                       generator_loss_final}

        # create the gif for the learning progress
        create_gif(self)

        # training has stopped
        self.train_status = "stop"

        # close the tensorflow session
        sess.close()
