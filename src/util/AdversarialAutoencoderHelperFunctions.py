"""
    Holds the functions shared by all three Autoencoders (Unsupervised, Supervised and SemiSupervised).
"""
import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.util import DataLoading


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
        any additional steps.
        """
        boundaries = [100000, 110000]
        values = [1.0, 0.5, 0.1]
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

    else:
        raise ValueError(decaying_learning_rate_name, "is not a valid value for this variable.")

    return learning_rate


def get_optimizer(aae_class, optimizer_name, sub_network_name, decaying_learning_rate_name=None, global_step=None):
    """
    wrapper function for the optimizers available in tensorflow. It returns the respective optimizer with the
    sub_network_name parameters stored in the AAE class with the specified decaying learning rate (if any).
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
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
                                                   getattr(aae_class, "learning_rate_" + sub_network_name))
    else:
        learning_rate = getattr(aae_class, "learning_rate_" + sub_network_name)

    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == "AdadeltaOptimizer":
        return tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate, rho=getattr(aae_class, "AdadeltaOptimizer_rho_" + sub_network_name),
            epsilon=getattr(aae_class, "AdadeltaOptimizer_epsilon_" + sub_network_name))
    elif optimizer_name == "AdagradOptimizer":
        return tf.train.AdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=getattr(aae_class, "AdagradOptimizer_initial_accumulator_value_" +
                                              sub_network_name))
    elif optimizer_name == "MomentumOptimizer":
        return tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=getattr(aae_class, "MomentumOptimizer_momentum_" + sub_network_name),
            use_nesterov=getattr(aae_class, "MomentumOptimizer_use_nesterov_" + sub_network_name))
    elif optimizer_name == "AdamOptimizer":
        return tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=getattr(aae_class, "AdamOptimizer_beta1_" + sub_network_name),
            beta2=getattr(aae_class, "AdamOptimizer_beta2_" + sub_network_name),
            epsilon=getattr(aae_class, "AdamOptimizer_epsilon_" + sub_network_name))
    elif optimizer_name == "FtrlOptimizer":
        return tf.train.FtrlOptimizer(
            learning_rate=learning_rate,
            learning_rate_power=getattr(aae_class, "FtrlOptimizer_learning_rate_power_" + sub_network_name),
            initial_accumulator_value=getattr(aae_class, "FtrlOptimizer_initial_accumulator_value_" +
                                              sub_network_name),
            l1_regularization_strength=getattr(aae_class, "FtrlOptimizer_l1_regularization_strength_" +
                                               sub_network_name),
            l2_regularization_strength=getattr(aae_class, "FtrlOptimizer_l2_regularization_strength_" +
                                               sub_network_name),
            l2_shrinkage_regularization_strength=
            getattr(aae_class, "FtrlOptimizer_l2_shrinkage_regularization_strength_" + sub_network_name)
        )
    elif optimizer_name == "ProximalGradientDescentOptimizer":
        return tf.train.ProximalGradientDescentOptimizer(
            learning_rate=learning_rate,
            l1_regularization_strength=getattr(aae_class, "ProximalGradientDescentOptimizer_l1_regularization_strength_"
                                               + sub_network_name),
            l2_regularization_strength=getattr(aae_class, "ProximalGradientDescentOptimizer_l2_regularization_strength_"
                                               + sub_network_name)
        )
    elif optimizer_name == "ProximalAdagradOptimizer":
        return tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=getattr(aae_class, "ProximalAdagradOptimizer_initial_accumulator_value_" +
                                              sub_network_name),
            l1_regularization_strength=getattr(aae_class, "ProximalAdagradOptimizer_l1_regularization_strength_" +
                                               sub_network_name),
            l2_regularization_strength=getattr(aae_class, "ProximalAdagradOptimizer_l2_regularization_strength_" +
                                               sub_network_name)
        )
    elif optimizer_name == "RMSPropOptimizer":
        return tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, decay=getattr(aae_class, "RMSPropOptimizer_decay_" + sub_network_name),
            momentum=getattr(aae_class, "RMSPropOptimizer_momentum_" + sub_network_name),
            epsilon=getattr(aae_class, "RMSPropOptimizer_epsilon_" + sub_network_name),
            centered=getattr(aae_class, "RMSPropOptimizer_centered_" + sub_network_name))


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


def get_input_data(selected_dataset):
    """
    returns the input data set based on self.selected_dataset
    :return: object holding the train data, the test data and the validation data
    """

    # hand written digits
    if selected_dataset == "MNIST":
        return DataLoading.read_mnist_data_from_ubyte('./data', one_hot=True)
    # Street View House Numbers
    elif selected_dataset == "SVHN":
        return DataLoading.read_svhn_from_mat('./data', one_hot=True, validation_size=5000)
    elif selected_dataset == "cifar10":
        return DataLoading.read_cifar10('./data', one_hot=True, validation_size=5000)
    elif selected_dataset == "custom":
        # TODO:
        print("not yet implemented")
        raise NotImplementedError


def draw_class_distribution_on_latent_space(latent_representations_current_epoch, labels_current_epoch, result_path, epoch):
    """
    draws the class distribution on the latent space.
    :param latent_representations_current_epoch: list of shape (n_batches*batch_size, z_dim) holding the encoder output,
    -> the latent representation of the inputs
    :param labels_current_epoch: list of shape (n_batches*batch_size, n_classes), holds the labels of the inputs
    encoded as one-hot vectors
    :param result_path: path where to store the resulting image
    :param: epoch: current epoch; for the file name of the resulting image
    :return:
    """

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

    plt.legend()
    plt.savefig(result_path + str(epoch) + "_latent_space_class_distribution" + '.png')
    plt.close('all')


def reshape_image_array(aae_class, img_array):
    """
    reshapes the image array based on the color scale:
        - gray scale: [input_dim_x, input_dim_y]
        - rgb scale: [input_dim_x, input_dim_y, 3]
    :param aae_class: AAE instance to access some important fields from it
    :param img_array: array/list of the image
    :return: reshaped np.array
    """

    # reshape the images according to the color scale
    if aae_class.color_scale == "gray_scale":
        # matplotlib wants a 2D array
        img = np.array(img_array).reshape(aae_class.input_dim_x, aae_class.input_dim_y)
    else:
        image_array = np.array(img_array)
        n_colored_pixels_per_channel = aae_class.input_dim_x * aae_class.input_dim_y
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
                                   batch_x, decoder_output, epoch, mini_batch_i):
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
    :return:
    """

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    plt.subplots(nrows=4, ncols=3)

    """
    plot the losses over time
    """
    plt.subplot(4, 3, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    plt.title("autoencoder_loss")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 2)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_losses"])
    plt.title("discriminator_loss")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["generator_losses"])
    plt.title("generator_loss")
    plt.xlabel("epoch")

    """
    plot the learning rates over time
    """
    plt.subplot(4, 3, 4)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 5)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_lr"])
    plt.title("discriminator_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 6)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    plt.title("generator_lr")
    plt.xlabel("epoch")

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
    plt.subplot(4, 3, 8)
    if aae_class.z_dim > 2:
        plt.hist(latent_representation.flatten())
    else:
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1])
    plt.title("encoder dist")

    # plot the discriminator outputs
    plt.subplot(4, 3, 9)
    plt.hist(discriminator_neg.flatten())
    plt.title("discriminator_neg dist")

    plt.subplot(4, 3, 10)
    plt.hist(discriminator_pos.flatten())
    plt.title("discriminator_pos dist")

    """
    plot one input image and its reconstruction
    """
    # plot one input image..
    plt.subplot(4, 3, 11)
    real_img = batch_x[0, :]
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(4, 3, 12)
    created_img = decoder_output[0, :]
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_minibatch_summary_image_semi_supervised(aae_class, real_dist, latent_representation, batch_x, decoder_output,
                                                   epoch, mini_batch_i, real_cat_dist, encoder_cat_dist):
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
    :return:
    """

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    plt.subplots(nrows=5, ncols=4)

    """
    plot the losses over time
    """
    plt.subplot(5, 4, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    plt.title("autoencoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 2)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_gaussian_losses"])
    plt.title("discriminator_gaussian_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_categorical_losses"])
    plt.title("discriminator_categorical_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 4)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["generator_losses"])
    plt.title("generator_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 5)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["supervised_encoder_loss"])
    plt.title("supervised_encoder_loss")
    plt.xlabel("epoch")

    """
    plot the learning rates over time
    """
    plt.subplot(5, 4, 6)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 7)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_g_lr"])
    plt.title("discriminator_g_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 8)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_c_lr"])
    plt.title("discriminator_c_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 9)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    plt.title("generator_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 10)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["supervised_encoder_lr"])
    plt.title("supervised_encoder_lr")
    plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """
    # plot one input image..
    plt.subplot(5, 4, 11)
    real_img = batch_x[0, :]
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(5, 4, 12)
    created_img = decoder_output[0, :]
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the accuracy
    """
    if len(aae_class.performance_over_time["accuracy"]) > 0:    # for the first epoch we don't not have accuracy yet
        plt.subplot(5, 4, 13)
        plt.plot(aae_class.performance_over_time["accuracy_epochs"],
                 aae_class.performance_over_time["accuracy"])
        plt.title("accuracy")
        plt.xlabel("epoch")

    """
    plot the encoder and the real categorical distribution 
    """
    plt.subplot(5, 4, 15)
    if aae_class.z_dim > 2:
        plt.hist(real_cat_dist.flatten())
    else:
        plt.scatter(real_cat_dist[:, 0], real_cat_dist[:, 1])
    plt.title("real cat dist")

    plt.subplot(5, 4, 16)
    if aae_class.z_dim > 2:
        plt.hist(encoder_cat_dist.flatten())
    else:
        plt.scatter(encoder_cat_dist[:, 0], encoder_cat_dist[:, 1])
    plt.title("encoder cat dist")

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(5, 4, 17)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    plt.subplot(5, 4, 18)
    if aae_class.z_dim > 2:
        plt.hist(latent_representation.flatten())
    else:
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1])
    plt.title("encoder dist")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def get_learning_rate_for_optimizer(optimizer):
    """
    returns the current learning rate for the provided optimizer
    :param optimizer: tf.train.Optimizer
    :return:
    """
    try:
        return optimizer._lr
    except AttributeError:
        return optimizer._learning_rate


# TODO: maybe generate image grid unsupervised
# TODO: maybe generate image grid (semi-)supervised
# TODO: maybe generate image grid z_dim

