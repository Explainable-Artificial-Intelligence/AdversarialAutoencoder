"""
    Holds the functions shared by all three Autoencoders (Unsupervised, Supervised and SemiSupervised).
"""
import glob

import imageio
import tensorflow as tf
import numpy as np
import datetime
import os
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.decomposition import PCA

from swagger_server.utils.Storage import Storage
from util import DataLoading


def use_activation_function_for_layer(activation_function, layer):
    """
    uses the provided activation function for the layer and returns the new tensor
    :param activation_function: ["relu", "relu6", "crelu", "elu", "softplus", "softsign", "sigmoid", "tanh", "linear"]
    :param layer: tensor of the layer of which the activation function should be applied to
    :return: tf tensor
    """
    if activation_function == "relu":
        return tf.nn.relu(layer)
    elif activation_function == "relu6":
        return tf.nn.relu6(layer)
    elif activation_function == "crelu":
        return tf.nn.crelu(layer)
    elif activation_function == "elu":
        return tf.nn.elu(layer)
    elif activation_function == "softplus":
        return tf.nn.softplus(layer)
    elif activation_function == "softsign":
        return tf.nn.softsign(layer)
    elif activation_function == "sigmoid":
        return tf.nn.sigmoid(layer)
    elif activation_function == "tanh":
        return tf.nn.tanh(layer)
    elif activation_function == "linear":
        return layer


def get_decaying_learning_rate(decaying_learning_rate_name, global_step, initial_learning_rate=0.1):
    """
    wrapper function for the decayed learning rate functions as defined in
    https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate
    :param decaying_learning_rate_name: ["exponential_decay", "inverse_time_decay", "natural_exp_decay",
    "piecewise_constant", "polynomial_decay"]
    :param global_step: A scalar int32 or int64 Tensor or a Python number. Global step to use for the decay
    computation. Must not be negative.
    :param initial_learning_rate: A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
    :return:
    """

    # TODO: params of different learning rates as kwargs

    if decaying_learning_rate_name == "exponential_decay":
        """
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        """
        decay_steps = 100000
        decay_rate = 0.96
        # If the argument staircase is True, then global_step / decay_steps is an integer division and the decayed
        # learning rate follows a staircase function.
        staircase = False
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                                   decay_steps, decay_rate, staircase)

    elif decaying_learning_rate_name == "inverse_time_decay":
        """
        staircase=False:
            decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
        staircase=True:
            decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
        """
        decay_steps = 1.0
        decay_rate = 0.5
        staircase = False
        learning_rate = tf.train.inverse_time_decay(initial_learning_rate, global_step,
                                                    decay_steps, decay_rate, staircase)

    elif decaying_learning_rate_name == "natural_exp_decay":
        """
        decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        """
        decay_steps = 1.0
        decay_rate = 0.5
        staircase = False
        learning_rate = tf.train.natural_exp_decay(initial_learning_rate, global_step,
                                                   decay_steps, decay_rate, staircase)

    elif decaying_learning_rate_name == "piecewise_constant":
        """
        Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5 for steps 100001 to 110000, and 0.1 for
        any additional steps:
            boundaries = [100000, 110000]
            values = [1.0, 0.5, 0.1]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        """
        boundaries = [250]
        values = [0.0001, 0.00001]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    elif decaying_learning_rate_name == "polynomial_decay":
        """
        cycle=False:
            global_step = min(global_step, decay_steps)
            decayed_learning_rate = (learning_rate - end_learning_rate) *
                                    (1 - global_step / decay_steps) ^ (power) +
                                    end_learning_rate
        cycle=True:
            decay_steps = decay_steps * ceil(global_step / decay_steps)
            decayed_learning_rate = (learning_rate - end_learning_rate) *
                                    (1 - global_step / decay_steps) ^ (power) +
                                    end_learning_rate
        """
        decay_steps = 10000
        end_learning_rate = 0.00001
        power = 1.0
        cycle = False
        learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step, decay_steps,
                                                  end_learning_rate, power, cycle)

    elif decaying_learning_rate_name == "static":
        """
        Static learning rate.
        """
        learning_rate = initial_learning_rate

    else:
        raise ValueError(decaying_learning_rate_name, "is not a valid value for this variable.")

    return learning_rate


def get_optimizer(param_dict, optimizer_name, sub_network_name, decaying_learning_rate_name=None, global_step=None):
    """
    wrapper function for the optimizers available in tensorflow. It returns the respective optimizer with the
    sub_network_name parameters stored in the AAE class with the specified decaying learning rate (if any).
    :param param_dict: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :param optimizer_name: name of the optimizer
    :param sub_network_name: one of ["autoencoder", "discriminator", "generator"]
    :param decaying_learning_rate_name: name of the decaying_learning_rate, if a decaying learning rate should be used,
    None otherwise
    :param global_step: A scalar int32 or int64 Tensor or a Python number. Global step to use for the decay
    computation. Must not be negative., None if no decaying learning rate should be used
    :return:
    """

    # if a decaying_learning_rate_name and global_step has been provided, use the decaying learning rate, otherwise
    # use a "normal" learning rate
    if decaying_learning_rate_name and global_step:
        learning_rate = get_decaying_learning_rate(decaying_learning_rate_name, global_step,
                                                   param_dict.get("learning_rate_" + sub_network_name))
    else:
        learning_rate = param_dict.get("learning_rate_" + sub_network_name)

    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == "AdadeltaOptimizer":
        return tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate, rho=param_dict.get("AdadeltaOptimizer_rho_" + sub_network_name),
            epsilon=param_dict.get("AdadeltaOptimizer_epsilon_" + sub_network_name))
    elif optimizer_name == "AdagradOptimizer":
        return tf.train.AdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=param_dict.get("AdagradOptimizer_initial_accumulator_value_" +
                                              sub_network_name))
    elif optimizer_name == "MomentumOptimizer":
        return tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=param_dict.get("MomentumOptimizer_momentum_" + sub_network_name),
            use_nesterov=param_dict.get("MomentumOptimizer_use_nesterov_" + sub_network_name))
    elif optimizer_name == "AdamOptimizer":
        return tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=param_dict.get("AdamOptimizer_beta1_" + sub_network_name),
            beta2=param_dict.get("AdamOptimizer_beta2_" + sub_network_name),
            epsilon=param_dict.get("AdamOptimizer_epsilon_" + sub_network_name))
    elif optimizer_name == "FtrlOptimizer":
        return tf.train.FtrlOptimizer(
            learning_rate=learning_rate,
            learning_rate_power=param_dict.get("FtrlOptimizer_learning_rate_power_" + sub_network_name),
            initial_accumulator_value=param_dict.get("FtrlOptimizer_initial_accumulator_value_" +
                                              sub_network_name),
            l1_regularization_strength=param_dict.get("FtrlOptimizer_l1_regularization_strength_" +
                                               sub_network_name),
            l2_regularization_strength=param_dict.get("FtrlOptimizer_l2_regularization_strength_" +
                                               sub_network_name),
            l2_shrinkage_regularization_strength=
            param_dict.get("FtrlOptimizer_l2_shrinkage_regularization_strength_" + sub_network_name)
        )
    elif optimizer_name == "ProximalGradientDescentOptimizer":
        return tf.train.ProximalGradientDescentOptimizer(
            learning_rate=learning_rate,
            l1_regularization_strength=param_dict.get("ProximalGradientDescentOptimizer_l1_regularization_strength_"
                                               + sub_network_name),
            l2_regularization_strength=param_dict.get("ProximalGradientDescentOptimizer_l2_regularization_strength_"
                                               + sub_network_name)
        )
    elif optimizer_name == "ProximalAdagradOptimizer":
        return tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=param_dict.get("ProximalAdagradOptimizer_initial_accumulator_value_" +
                                              sub_network_name),
            l1_regularization_strength=param_dict.get("ProximalAdagradOptimizer_l1_regularization_strength_" +
                                               sub_network_name),
            l2_regularization_strength=param_dict.get("ProximalAdagradOptimizer_l2_regularization_strength_" +
                                               sub_network_name)
        )
    elif optimizer_name == "RMSPropOptimizer":
        return tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, decay=param_dict.get("RMSPropOptimizer_decay_" + sub_network_name),
            momentum=param_dict.get("RMSPropOptimizer_momentum_" + sub_network_name),
            epsilon=param_dict.get("RMSPropOptimizer_epsilon_" + sub_network_name),
            centered=param_dict.get("RMSPropOptimizer_centered_" + sub_network_name))


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
        return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


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


def reshape_tensor_to_rgb_image(image_array, input_dim_x, input_dim_y):
    """
    reshapes the given image array to an rgb image for the tensorboard visualization
    :param image_array: array of images to be reshaped
    :param input_dim_x: dim x of the images
    :param input_dim_y: dim y of the images
    :return: rgb array of the image
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


def form_results(aae_class):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three strings pointing to tensorboard, saved models and log paths respectively.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", ":").replace(":", "_")

    folder_name = "/{0}_{1}".format(date, aae_class.selected_dataset)
    aae_class.result_folder_name = folder_name
    tensorboard_path = aae_class.results_path + folder_name + '/Tensorboard'
    saved_model_path = aae_class.results_path + folder_name + '/Saved_models/'
    log_path = aae_class.results_path + folder_name + '/log'

    if not os.path.exists(aae_class.results_path):
        os.mkdir(aae_class.results_path)
    if not os.path.exists(aae_class.results_path + folder_name):
        os.mkdir(aae_class.results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def get_input_data(selected_dataset, color_scale):
    """
    returns the input data set based on self.selected_dataset
    :return: object holding the train data, the test data and the validation data
    """
    return DataLoading.get_input_data(selected_dataset, color_scale=color_scale)


def draw_class_distribution_on_latent_space(latent_representations_current_epoch, labels_current_epoch, result_path,
                                            epoch, random_points_for_image_grid, combined_plot=False):
    """
    draws the class distribution on the latent space.
    :param latent_representations_current_epoch: list of shape (n_batches*batch_size, z_dim) holding the encoder output,
    -> the latent representation of the inputs
    :param labels_current_epoch: list of shape (n_batches*batch_size, n_classes), holds the labels of the inputs
    encoded as one-hot vectors
    :param result_path: path where to store the resulting image
    :param epoch: current epoch; for the file name of the resulting image
    :param random_points_for_image_grid: random points used for generating the image grid (only for (semi-) supervised
    autoencoders)
    :param combined_plot: whether the plot should be combined with the image grid
    :return:
    """

    if combined_plot:
        plt.subplot(1, 2, 2)

    # convert lists to numpy array
    latent_representations_current_epoch = np.array(latent_representations_current_epoch)
    labels_current_epoch = np.array(labels_current_epoch)

    # convert one hot vectors to integer labels
    int_labels = np.argmax(labels_current_epoch, axis=1)

    # get the dimension of the latent space and the number of classes
    z_dim = latent_representations_current_epoch.shape[1]
    n_classes = labels_current_epoch.shape[1]

    # perform PCA if the dimension of the latent space is higher than 2
    if z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representations_current_epoch)
        latent_representations_current_epoch = pca.transform(latent_representations_current_epoch)

    # plot the different classes on the latent space
    for class_label in range(n_classes):
        # get the points corresponding to the same classes
        points_for_current_class_label = latent_representations_current_epoch[np.where(int_labels == class_label)]
        # plot them
        plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1], label=str(class_label))

    # draw the random points on the latent space if we have any
    if random_points_for_image_grid is not None:

        # markers for the random points
        markers = ["^", "v", ">", "<", "s", "*", "_", "+", "x", "|"]

        for random_point, marker in zip(random_points_for_image_grid, markers):
            # TODO: pca
            # pca.transform()
            plt.scatter(random_point[0], random_point[1], marker=marker, c="black")

    if combined_plot:
        plt.suptitle("Epoch: " + str(epoch))

    plt.legend()
    plt.savefig(result_path + str(epoch) + "_latent_space_class_distribution" + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def reshape_image_array(aae_class, img_array, is_array_of_arrays=False):
    """
    reshapes the image array based on the color scale:
        - gray scale: [input_dim_x, input_dim_y]
        - rgb scale: [input_dim_x, input_dim_y, 3]
    :param aae_class: AAE instance to access some important fields from it
    :param img_array: array/list of the image
    :param is_array_of_arrays: whether we have an array of arrays or not
    :return: reshaped np.array
    """

    # reshape the images according to the color scale
    if aae_class.color_scale == "gray_scale":
        # matplotlib wants a 2D array
        img = np.array(img_array).reshape(aae_class.input_dim_x, aae_class.input_dim_y)
    else:
        image_array = np.array(img_array)
        n_colored_pixels_per_channel = aae_class.input_dim_x * aae_class.input_dim_y

        if is_array_of_arrays:
            # first n_colored_pixels_per_channel encode red
            red_pixels = image_array[:, :n_colored_pixels_per_channel].reshape(aae_class.input_dim_x,
                                                                               aae_class.input_dim_y, 1)
            # next n_colored_pixels_per_channel encode green
            green_pixels = image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                .reshape(aae_class.input_dim_x, aae_class.input_dim_y, 1)
            # last n_colored_pixels_per_channel encode blue
            blue_pixels = image_array[:, n_colored_pixels_per_channel * 2:].reshape(aae_class.input_dim_x,
                                                                                    aae_class.input_dim_y, 1)
        else:
            # first n_colored_pixels_per_channel encode red
            red_pixels = image_array[:n_colored_pixels_per_channel].reshape(aae_class.input_dim_x,
                                                                            aae_class.input_dim_y,
                                                                            1)
            # next n_colored_pixels_per_channel encode green
            green_pixels = image_array[n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                .reshape(aae_class.input_dim_x, aae_class.input_dim_y, 1)
            # last n_colored_pixels_per_channel encode blue
            blue_pixels = image_array[n_colored_pixels_per_channel * 2:].reshape(
                aae_class.input_dim_x,
                aae_class.input_dim_y, 1)

        # concatenate the color arrays into one array
        img = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)

    return img


def create_minibatch_summary_image(aae_class, real_dist, latent_representation, discriminator_neg, discriminator_pos,
                                   batch_x, decoder_output, epoch, mini_batch_i, batch_labels,
                                   include_tuning_performance=False):
    """
    creates a summary image displaying the losses and the learning rates over time, the real distribution and the latent
    representation, the discriminator outputs (pos and neg) and one input image and its reconstruction image
    :param aae_class: AAE instance to access some important fields from it
    :param real_dist: real distribution the AAE should map to
    :param latent_representation: latent representation of the AAE
    :param discriminator_neg: output of the discriminator for the negative samples q(z) (generated by the generator)
    :param discriminator_pos: output of the discriminator for the positive samples p(z) (from a real data distribution)
    :param batch_x: current batch of input images
    :param decoder_output: output of the decoder for the current batch
    :param epoch: current epoch of the training (only used for the image filename)
    :param mini_batch_i: current iteration of the minibatch (only used for the image filename)
    :param batch_labels: list of shape (n_batches*batch_size, n_classes), holds the labels of the inputs
    encoded as one-hot vectors
    :param include_tuning_performance: whether to include the losses and learning rates from the other adversarial
    autoencoders in the same tuning process in the plots
    :return:
    """

    # convert lists to numpy array
    latent_representation = np.array(latent_representation)
    batch_labels = np.array(batch_labels)

    # convert one hot vectors to integer labels
    batch_integer_labels = np.argmax(batch_labels, axis=1)

    # get the number of classes
    n_classes = batch_labels.shape[1]

    # calculate the total losses
    total_losses = [sum(x) for x in zip(aae_class.performance_over_time["autoencoder_losses"],
                                        aae_class.performance_over_time["discriminator_losses"],
                                        aae_class.performance_over_time["generator_losses"])]

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=4, ncols=3)

    """
    plot the losses over time
    """
    plt.subplot(4, 3, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["autoencoder_losses"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder loss")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 2)
    plt.stackplot(aae_class.performance_over_time["list_of_epochs"],
                  [aae_class.performance_over_time["discriminator_losses"],
                   aae_class.performance_over_time["generator_losses"]], labels=["discriminator loss",
                                                                                 "generator loss"])
    plt.title("discriminator-generator ratio")
    plt.legend()
    plt.xlabel("epoch")

    plt.subplot(4, 3, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"], total_losses)
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            # calculate the total losses
            total_loss_current_aae = [sum(x) for x in zip(performance_dict["autoencoder_losses"],
                                                          performance_dict["discriminator_losses"],
                                                          performance_dict["generator_losses"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot], total_loss_current_aae[:n_points_to_plot],
                     alpha=0.5, c="gray")
    plt.title("total loss")
    plt.xlabel("epoch")

    # usually we use a stack plot for the discriminator and generator loss, but this won't work for several results,
    # so we use our currently unused subplot to plot the losses in a line chart
    if include_tuning_performance:
        plt.subplot(4, 3, 4)
        discriminator_generator_loss = [sum(x) for x in zip(aae_class.performance_over_time["discriminator_losses"],
                                                            aae_class.performance_over_time["generator_losses"])]
        plt.plot(aae_class.performance_over_time["list_of_epochs"], discriminator_generator_loss)
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            # calculate the total losses
            discriminator_generator_loss_current_aae = [sum(x) for x in zip(performance_dict["discriminator_losses"],
                                                                            performance_dict["generator_losses"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     discriminator_generator_loss_current_aae[:n_points_to_plot], alpha=0.5, c="gray")
        plt.title("generator + discriminator loss")
        plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """
    # plot one input image..
    plt.subplot(4, 3, 5)
    real_img = batch_x[0, :]
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(4, 3, 6)
    created_img = decoder_output[0, :]
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(4, 3, 7)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    # TODO: pca for z_dim > 2
    plt.subplot(4, 3, 8)
    if aae_class.z_dim > 2:
        plt.hist(latent_representation.flatten())
    else:
        # plot the different classes on the latent space
        for class_label in range(n_classes):
            # get the points corresponding to the same classes
            points_for_current_class_label = latent_representation[np.where(batch_integer_labels == class_label)]
            # plot them
            plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1],
                        label=str(class_label))
    plt.legend()
    plt.title("encoder dist")

    # plot the discriminator outputs
    plt.subplot(4, 3, 9)
    plt.hist(discriminator_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.legend()
    plt.title("discriminator distr.")

    """
    plot the learning rates over time
    """
    plt.subplot(4, 3, 10)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["autoencoder_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 11)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["discriminator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("discriminator_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 12)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["generator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_lr")
    plt.xlabel("epoch")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_minibatch_summary_image_semi_supervised(aae_class, real_dist, latent_representation, batch_x, decoder_output,
                                                   epoch, mini_batch_i, real_cat_dist, encoder_cat_dist, batch_labels,
                                                   discriminator_gaussian_neg, discriminator_gaussian_pos,
                                                   discriminator_cat_neg, discriminator_cat_pos,
                                                   include_tuning_performance=False):
    """
    creates a summary image displaying the losses and the learning rates over time, the real distribution and the latent
    representation, the discriminator outputs (pos and neg) and one input image and its reconstruction image
    :param aae_class: AAE instance to access some important fields from it
    :param real_dist: real distribution the AAE should map to
    :param latent_representation: latent representation of the AAE
    :param batch_x: current batch of input images
    :param decoder_output: output of the decoder for the current batch
    :param epoch: current epoch of the training (only used for the image filename)
    :param mini_batch_i: current iteration of the minibatch (only used for the image filename)
    :param real_cat_dist: real categorical distribution
    :param encoder_cat_dist: encoder output of the categorical distribution
    :param include_tuning_performance: whether to include the losses and learning rates from the other adversarial
    autoencoders in the same tuning process in the plots
    :return:
    """

    # convert lists to numpy array
    latent_representation = np.array(latent_representation)
    batch_labels = np.array(batch_labels)

    # convert one hot vectors to integer labels
    batch_integer_labels = np.argmax(batch_labels, axis=1)
    real_cat_dist_integer_labels = np.argmax(real_cat_dist, axis=1)
    encoder_cat_dist_integer_labels = np.argmax(encoder_cat_dist, axis=1)

    # get the number of classes
    n_classes = batch_labels.shape[1]

    # calculate the total losses
    total_losses = [sum(x) for x in zip(aae_class.performance_over_time["autoencoder_losses"],
                                        aae_class.performance_over_time["discriminator_gaussian_losses"],
                                        aae_class.performance_over_time["discriminator_categorical_losses"],
                                        aae_class.performance_over_time["generator_losses"],
                                        aae_class.performance_over_time["supervised_encoder_loss"])]

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=5, ncols=4)

    """
    plot the losses over time
    """
    plt.subplot(5, 4, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            plt.plot(performance_dict["list_of_epochs"], performance_dict["autoencoder_losses"], alpha=0.5, c="gray")
    plt.title("autoencoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 2)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_gaussian_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_gaussian_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_gaussian_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_categorical_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_categorical_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_categorical_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 4)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["generator_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["generator_losses"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 5)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["supervised_encoder_loss"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["supervised_encoder_loss"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("supervised_encoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 6)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             total_losses)
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            total_loss_current_aae = [sum(x) for x in zip(performance_dict["autoencoder_losses"],
                                                          performance_dict["discriminator_gaussian_losses"],
                                                          performance_dict["discriminator_categorical_losses"],
                                                          performance_dict["generator_losses"],
                                                          performance_dict["supervised_encoder_loss"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     total_loss_current_aae[:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("total loss")
    plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """

    # # plot one input image..
    plt.subplot(5, 4, 7)
    real_img = batch_x[0, :]
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(5, 4, 8)
    created_img = decoder_output[0, :]
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the accuracy
    """
    if len(aae_class.performance_over_time["accuracy"]) > 0:  # for the first epoch we don't not have accuracy yet
        plt.subplot(5, 4, 9)
        plt.plot(aae_class.performance_over_time["accuracy_epochs"],
                 aae_class.performance_over_time["accuracy"])
        # plot the performance of the other adv autoencoders
        if include_tuning_performance:
            # get the dictionary holding the dictionaries which hold the performance over time
            tuning_performances = Storage.get_tuning_results_performance_over_time()
            for key, performance_dict in tuning_performances.items():
                n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
                plt.plot(performance_dict["accuracy_epochs"][:n_points_to_plot],
                         performance_dict["accuracy"][:n_points_to_plot], alpha=0.5, c="gray")
        plt.title("accuracy")
        plt.xlabel("epoch")

    """
    plot the batch labels and the predicted labels
    """
    plt.subplot(5, 4, 10)
    plt.hist(batch_integer_labels, alpha=0.5, label="batch labels", color="#d95f02")
    plt.hist(encoder_cat_dist_integer_labels, alpha=0.5, label="predicted labels", color="#1b9e77")
    plt.legend()
    plt.title("label prediction")

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(5, 4, 11)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    # TODO: pca for z_dim > 2
    plt.subplot(5, 4, 12)
    if aae_class.z_dim > 2:
        plt.hist(latent_representation.flatten())
    else:
        # plot the different classes on the latent space
        for class_label in range(n_classes):
            # get the points corresponding to the same classes
            points_for_current_class_label = latent_representation[np.where(batch_integer_labels == class_label)]
            # plot them
            plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1],
                        label=str(class_label))
    plt.legend()
    plt.title("encoder dist")

    """
    plot the discriminator outputs
    """
    plt.subplot(5, 4, 13)
    plt.hist(discriminator_gaussian_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_gaussian_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator gaussian")
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.hist(discriminator_cat_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_cat_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator categorical")
    plt.legend()

    """
    plot the learning rates over time
    """
    plt.subplot(5, 4, 16)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["autoencoder_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 17)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_g_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["discriminator_g_lr"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_g_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 18)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_c_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"], learning_rate_dict["discriminator_c_lr"], alpha=0.5,
                     c="gray")
    plt.title("discriminator_c_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 19)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["generator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 20)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["supervised_encoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["supervised_encoder_lr"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("supervised_encoder_lr")
    plt.xlabel("epoch")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def get_learning_rate_for_optimizer(optimizer, sess):
    """
    returns the current learning rate for the provided optimizer
    :param optimizer: tf.train.Optimizer
    :param sess: tf.session
    :return:
    """
    try:
        if isinstance(optimizer._lr , float):
            return optimizer._lr
        else:
            return sess.run(optimizer._lr )
    except AttributeError:
        if isinstance(optimizer._learning_rate , float):
            return optimizer._learning_rate
        else:
            return sess.run(optimizer._lr )


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    :param text: text sorted in human order ()
    :return:
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def create_gif(aae_class):
    """
    creates a gif showing the learning progress on the latent space and the class distribution on it
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :return:
    """
    result_path = aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/'
    filenames = glob.glob(result_path + "*_latent_space_class_distribution.png")
    filenames.sort(key=natural_keys)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    if len(images) > 0:
        imageio.mimwrite(result_path + 'latent_space_class_distribution.gif', images, duration=1.0)


def get_layer_names(aae_class):
    """
    returns all layer names for the given autoencoder as dictionary with the subnetwork (encoder, decoder,
    discriminator, etc.) as key and a list of layer names as value
    :param aae_class: AAE instance to access some important fields from it
    :return: dictionary with the subnetwork (encoder, decoder, discriminator, etc.) as key and a list of layer names
    as value
    """

    # get the autoencoder class ["UnsupervisedAdversarialAutoencoder", "SupervisedAdversarialAutoencoder",
    # "SemiSupervisedAdversarialAutoencoder"]
    selected_autoencoder = aae_class.__class__.__name__

    # get the layer names for the encoder and decoder, since those are present for all of the autoencoders
    encoder_layer_names = ["encoder_dense_layer_" + str(i)
                           for i in range(1, len(aae_class.n_neurons_of_hidden_layer_x_autoencoder) + 1)]
    encoder_layer_names += ["encoder_output"]

    decoder_layer_names = ["decoder_dense_layer_" + str(i)
                           for i in range(1, len(aae_class.n_neurons_of_hidden_layer_x_autoencoder) + 1)]
    decoder_layer_names += ["decoder_output"]

    # create a dictionary holding the layer names
    all_layer_names = {"encoder": encoder_layer_names, "decoder": decoder_layer_names}

    # get the autoencoder specific layer names
    if selected_autoencoder == "SemiSupervisedAdversarialAutoencoder":
        discriminator_gaussian_layer_names = ["discriminator_gaussian_dense_layer_" + str(i) for i in
                                              range(1, len(aae_class.n_neurons_of_hidden_layer_x_discriminator_g) + 1)]
        discriminator_gaussian_layer_names += ["discriminator_gaussian_output"]

        discriminator_categorical_layer_names = ["discriminator_categorical_dense_layer_" + str(i) for i in
                                              range(1, len(aae_class.n_neurons_of_hidden_layer_x_discriminator_c) + 1)]
        discriminator_categorical_layer_names += ["discriminator_categorical_output"]

        # add them to the dictionary
        all_layer_names["discriminator_gaussian"] = discriminator_gaussian_layer_names
        all_layer_names["discriminator_categorical"] = discriminator_categorical_layer_names

    # (Un)-Supervised autoencoder
    else:
        discriminator_layer_names = ["discriminator_dense_layer_" + str(i) for i in
                                     range(1, len(aae_class.n_neurons_of_hidden_layer_x_discriminator) + 1)]
        discriminator_layer_names += ["discriminator_output"]

        # add them to the dictionary
        all_layer_names["discriminator"] = discriminator_layer_names

    return all_layer_names


def get_all_layer_vars(aae_class, var_to_get):
    """
    returns either the weights or the biases as dictionary with the layer_name as key and the weights/biases as value
    :param aae_class: AAE instance to access some important fields from it
    :param var_to_get: one of ["weights", "bias"]
    :return: dictionary with the layer_name as key and the weights/biases as value
    """

    if var_to_get not in ["weights", "bias"]:
        raise ValueError

    # holds the weights/biases for the layers
    all_vars = {}

    # iterate over the different parts of the aae ("encoder", "decoder", "discriminator")
    for subnetwork in aae_class.all_layer_names:
        # iterate over the layers of each part
        for layer_name in aae_class.all_layer_names[subnetwork]:
            current_layer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=layer_name)
            # iterate over the variables stored for each layer (weight, bias, several variables for
            # the AdamOptimizer, if it is used
            for variable in current_layer:
                # we're only interested in the weights
                if variable.name == layer_name + "/" + var_to_get + ":0":
                    # get the values
                    var_values = variable.eval()
                    # store it in all_vars dict
                    all_vars[layer_name] = var_values

    return all_vars


def get_biases_or_weights_for_layer(aae_class, function_params):
    """
    returns either the weights or the biases for the layer specified in the function params
    :param aae_class:
    :param sess:
    :param function_params:
    :return:
    """

    bias_or_weights, layer_name = function_params
    all_values = None
    if bias_or_weights == "bias":
        all_values = get_all_layer_vars(aae_class, "bias")
    elif bias_or_weights == "weights":
        all_values = get_all_layer_vars(aae_class, "weights")

    return all_values[layer_name]


def visualize_autoencoder_weights_and_biases(aae_class, epoch):
    """
    visualizes the autoencoder weights and biases by drawing histograms of them and plotting the weights of the input
    layer
    :param aae_class: AAE instance to access some important fields from it
    :param epoch: epoch we currently in (for the filename)
    :return:
    """

    # get all the weights
    all_weights = get_all_layer_vars(aae_class, "weights")

    # get all the bias
    all_biases = get_all_layer_vars(aae_class, "bias")

    # create the path where the weight images should be stored
    result_path_weight_img = aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + \
                             str(epoch) + "_"

    # visualize the weights
    visualize_single_layer_with_histogram(all_weights, "encoder_dense_layer_1", "weights", result_path_weight_img,
                                          epoch)

    # visualize the biases
    visualize_single_layer_with_histogram(all_biases, "encoder_dense_layer_1", "bias", result_path_weight_img, epoch)

    # visualize the input layer weights
    input_layer_weights = all_weights["encoder_dense_layer_1"]
    visualize_input_layer_weights(input_layer_weights, result_path_weight_img, aae_class.input_dim_x,
                                  aae_class.input_dim_y, aae_class.color_scale, epoch)


def visualize_input_layer_weights(weights, result_path, input_dim_x, input_dim_y, color_scale, epoch):
    """
    visualizes the weights of the input layer of the encoder by plotting them on one image (gray scale) or three
    separate images for each channel (rgb scale)
    :param weights: numpy array of the weights with shape (input_dim, n_neurons_first_layer_encoder)
    :param result_path: the path for saving the image(s)
    :param input_dim_x: input dimension for x direction
    :param input_dim_y: input dimension for y direction
    :param color_scale: which color scale the data uses ("gray_scale" or "rgb_scale")
    :param epoch: current epoch (for the plot title)
    :return:
    """

    # calculate the mean weights for each input pixel
    means_per_row = np.mean(weights, axis=1)

    if color_scale == "gray_scale":
        # reshape the array for matplotlib
        means_per_row_img = means_per_row.reshape(input_dim_x, input_dim_y)

        # plot the array and save it
        plt.imshow(means_per_row_img, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights.png")
        plt.close()
    else:
        # get the number of colored pixels for each channel
        n_colored_pixels_per_channel = input_dim_x * input_dim_y

        # first n_colored_pixels_per_channel encode red
        red_pixels = means_per_row[:n_colored_pixels_per_channel].reshape(input_dim_x, input_dim_y)
        # next n_colored_pixels_per_channel encode green
        green_pixels = means_per_row[n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2].reshape(input_dim_x,
                                                                                                            input_dim_y)
        # last n_colored_pixels_per_channel encode blue
        blue_pixels = means_per_row[n_colored_pixels_per_channel * 2:].reshape(input_dim_x, input_dim_y)

        # plot the arrays for the different channels and save it
        plt.imshow(red_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_red_channel.png")
        plt.close()

        plt.imshow(green_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_green_channel.png")
        plt.close()

        plt.imshow(blue_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_blue_channel.png")
        plt.close()


def visualize_single_layer_with_histogram(all_values_for_layers, layer_name, var_to_visualize, result_path, epoch):
    """
    visualizes the layer variables (weights or biases) with a histogram
    :param all_values_for_layers: numpy array holding the weights or the biases of all layers
    :param layer_name: name of the layer to visualize
    :param var_to_visualize: one of ["weights", "bias"]
    :param result_path: path for saving the images to
    :param epoch: current epoch (for the plot title)
    :return:
    """

    value_for_layer = all_values_for_layers[layer_name]

    print("visualize_single_layer_with_histogram")
    print(value_for_layer.shape)

    plt.hist(value_for_layer.flatten())
    plt.title("Epoch: " + str(epoch))
    plt.savefig(result_path + layer_name + "_" + var_to_visualize + ".png")
    plt.close()


def visualize_weights_single_layer_as_img_grid(weights, input_dim, result_path_weight_img):

    # TODO: color scale!
    #

    # we assume quadratic input images
    nx = int(np.sqrt(input_dim))

    # increase figure size
    plt.rcParams["figure.figsize"] = (30, 40)

    plt.subplot()
    gs = gridspec.GridSpec(nx, nx, hspace=0.05, wspace=0.05)

    for i, grid in enumerate(gs):

        weight = weights[i]

        # we want a quadratic image..
        # TODO: deal with hard coded stuff
        weight = weight[:961]

        weight_img = np.reshape(weight, (31, 31))

        # img = np.array(img_array).reshape(aae_class.input_dim_x, aae_class.input_dim_y)

        # print(i)

        ax = plt.subplot(gs[i])
        ax.imshow(weight_img, cmap="gray")

    plt.savefig(result_path_weight_img)

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

# TODO: maybe generate image grid unsupervised
# TODO: maybe generate image grid (semi-)supervised
# TODO: maybe generate image grid z_dim

