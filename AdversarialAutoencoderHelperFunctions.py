import tensorflow as tf
import datetime
import os

import DataLoading


def get_optimizer_autoencoder(aae_class, optimizer_name):
    """
    wrapper function for the optimizers available in tensorflow. It returns the respective optimizer with the
    autoencoder parameters stored in the AAE class.
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :param optimizer_name: name of the optimizer
    :return: respective tensorflow optimizer
    """
    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate=aae_class.learning_rate_autoencoder)
    elif optimizer_name == "AdadeltaOptimizer":
        return tf.train.AdadeltaOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder, rho=aae_class.AdadeltaOptimizer_rho_autoencoder,
            epsilon=aae_class.AdadeltaOptimizer_epsilon_autoencoder)
    elif optimizer_name == "AdagradOptimizer":
        return tf.train.AdagradOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder,
            initial_accumulator_value=aae_class.AdagradOptimizer_initial_accumulator_value_autoencoder),
    elif optimizer_name == "MomentumOptimizer":
        return tf.train.MomentumOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder, momentum=aae_class.MomentumOptimizer_momentum_autoencoder,
            use_nesterov=aae_class.MomentumOptimizer_use_nesterov_autoencoder)
    elif optimizer_name == "AdamOptimizer":
        return tf.train.AdamOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder, beta1=aae_class.AdamOptimizer_beta1_autoencoder,
            beta2=aae_class.AdamOptimizer_beta2_autoencoder, epsilon=aae_class.AdamOptimizer_epsilon_autoencoder)
    elif optimizer_name == "FtrlOptimizer":
        return tf.train.FtrlOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder,
            learning_rate_power=aae_class.FtrlOptimizer_learning_rate_power_autoencoder,
            initial_accumulator_value=aae_class.FtrlOptimizer_initial_accumulator_value_autoencoder,
            l1_regularization_strength=aae_class.FtrlOptimizer_l1_regularization_strength_autoencoder,
            l2_regularization_strength=aae_class.FtrlOptimizer_l2_regularization_strength_autoencoder,
            l2_shrinkage_regularization_strength=aae_class.FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder
        )
    elif optimizer_name == "ProximalGradientDescentOptimizer":
        return tf.train.ProximalGradientDescentOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder,
            l1_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder,
            l2_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder
        )
    elif optimizer_name == "ProximalAdagradOptimizer":
        return tf.train.ProximalAdagradOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder,
            initial_accumulator_value=aae_class.ProximalAdagradOptimizer_initial_accumulator_value_autoencoder,
            l1_regularization_strength=aae_class.ProximalAdagradOptimizer_l1_regularization_strength_autoencoder,
            l2_regularization_strength=aae_class.ProximalAdagradOptimizer_l2_regularization_strength_autoencoder
        )
    elif optimizer_name == "RMSPropOptimizer":
        return tf.train.RMSPropOptimizer(
            learning_rate=aae_class.learning_rate_autoencoder, decay=aae_class.RMSPropOptimizer_decay_autoencoder,
            momentum=aae_class.RMSPropOptimizer_momentum_autoencoder, epsilon=aae_class.RMSPropOptimizer_epsilon_autoencoder,
            centered=aae_class.RMSPropOptimizer_centered_autoencoder)


def get_optimizer_discriminator(aae_class, optimizer_name):
    """
    wrapper function for the optimizers available in tensorflow. It returns the respective optimizer with the
    discriminator parameters stored in the AAE class.
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :param optimizer_name: name of the optimizer
    :return: respective tensorflow optimizer
    """
    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate=aae_class.learning_rate_discriminator)
    elif optimizer_name == "AdadeltaOptimizer":
        return tf.train.AdadeltaOptimizer(
            learning_rate=aae_class.learning_rate_discriminator, rho=aae_class.AdadeltaOptimizer_rho_discriminator,
            epsilon=aae_class.AdadeltaOptimizer_epsilon_discriminator)
    elif optimizer_name == "AdagradOptimizer":
        return tf.train.AdagradOptimizer(
            learning_rate=aae_class.learning_rate_discriminator,
            initial_accumulator_value=aae_class.AdagradOptimizer_initial_accumulator_value_discriminator),
    elif optimizer_name == "MomentumOptimizer":
        return tf.train.MomentumOptimizer(
            learning_rate=aae_class.learning_rate_discriminator, momentum=aae_class.MomentumOptimizer_momentum_discriminator,
            use_nesterov=aae_class.MomentumOptimizer_use_nesterov_discriminator)
    elif optimizer_name == "AdamOptimizer":
        return tf.train.AdamOptimizer(
            learning_rate=aae_class.learning_rate_discriminator, beta1=aae_class.AdamOptimizer_beta1_discriminator,
            beta2=aae_class.AdamOptimizer_beta2_discriminator, epsilon=aae_class.AdamOptimizer_epsilon_discriminator)
    elif optimizer_name == "FtrlOptimizer":
        return tf.train.FtrlOptimizer(
            learning_rate=aae_class.learning_rate_discriminator,
            learning_rate_power=aae_class.FtrlOptimizer_learning_rate_power_discriminator,
            initial_accumulator_value=aae_class.FtrlOptimizer_initial_accumulator_value_discriminator,
            l1_regularization_strength=aae_class.FtrlOptimizer_l1_regularization_strength_discriminator,
            l2_regularization_strength=aae_class.FtrlOptimizer_l2_regularization_strength_discriminator,
            l2_shrinkage_regularization_strength=aae_class.FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator
        )
    elif optimizer_name == "ProximalGradientDescentOptimizer":
        return tf.train.ProximalGradientDescentOptimizer(
            learning_rate=aae_class.learning_rate_discriminator,
            l1_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator,
            l2_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator
        )
    elif optimizer_name == "ProximalAdagradOptimizer":
        return tf.train.ProximalAdagradOptimizer(
            learning_rate=aae_class.learning_rate_discriminator,
            initial_accumulator_value=aae_class.ProximalAdagradOptimizer_initial_accumulator_value_discriminator,
            l1_regularization_strength=aae_class.ProximalAdagradOptimizer_l1_regularization_strength_discriminator,
            l2_regularization_strength=aae_class.ProximalAdagradOptimizer_l2_regularization_strength_discriminator
        )
    elif optimizer_name == "RMSPropOptimizer":
        return tf.train.RMSPropOptimizer(
            learning_rate=aae_class.learning_rate_discriminator, decay=aae_class.RMSPropOptimizer_decay_discriminator,
            momentum=aae_class.RMSPropOptimizer_momentum_discriminator,
            epsilon=aae_class.RMSPropOptimizer_epsilon_discriminator,
            centered=aae_class.RMSPropOptimizer_centered_discriminator)


def get_optimizer_generator(aae_class, optimizer_name):
    """
    wrapper function for the optimizers available in tensorflow. It returns the respective optimizer with the
    generator parameters stored in the AAE class.
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :param optimizer_name: name of the optimizer
    :return: respective tensorflow optimizer
    """
    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate=aae_class.learning_rate_generator)
    elif optimizer_name == "AdadeltaOptimizer":
        return tf.train.AdadeltaOptimizer(
            learning_rate=aae_class.learning_rate_generator, rho=aae_class.AdadeltaOptimizer_rho_generator,
            epsilon=aae_class.AdadeltaOptimizer_epsilon_generator)
    elif optimizer_name == "AdagradOptimizer":
        return tf.train.AdagradOptimizer(
            learning_rate=aae_class.learning_rate_generator,
            initial_accumulator_value=aae_class.AdagradOptimizer_initial_accumulator_value_generator),
    elif optimizer_name == "MomentumOptimizer":
        return tf.train.MomentumOptimizer(
            learning_rate=aae_class.learning_rate_generator, momentum=aae_class.MomentumOptimizer_momentum_generator,
            use_nesterov=aae_class.MomentumOptimizer_use_nesterov_generator)
    elif optimizer_name == "AdamOptimizer":
        return tf.train.AdamOptimizer(
            learning_rate=aae_class.learning_rate_generator, beta1=aae_class.AdamOptimizer_beta1_generator,
            beta2=aae_class.AdamOptimizer_beta2_generator, epsilon=aae_class.AdamOptimizer_epsilon_generator)
    elif optimizer_name == "FtrlOptimizer":
        return tf.train.FtrlOptimizer(
            learning_rate=aae_class.learning_rate_generator,
            learning_rate_power=aae_class.FtrlOptimizer_learning_rate_power_generator,
            initial_accumulator_value=aae_class.FtrlOptimizer_initial_accumulator_value_generator,
            l1_regularization_strength=aae_class.FtrlOptimizer_l1_regularization_strength_generator,
            l2_regularization_strength=aae_class.FtrlOptimizer_l2_regularization_strength_generator,
            l2_shrinkage_regularization_strength=aae_class.FtrlOptimizer_l2_shrinkage_regularization_strength_generator
        )
    elif optimizer_name == "ProximalGradientDescentOptimizer":
        return tf.train.ProximalGradientDescentOptimizer(
            learning_rate=aae_class.learning_rate_generator,
            l1_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l1_regularization_strength_generator,
            l2_regularization_strength=aae_class.ProximalGradientDescentOptimizer_l2_regularization_strength_generator
        )
    elif optimizer_name == "ProximalAdagradOptimizer":
        return tf.train.ProximalAdagradOptimizer(
            learning_rate=aae_class.learning_rate_generator,
            initial_accumulator_value=aae_class.ProximalAdagradOptimizer_initial_accumulator_value_generator,
            l1_regularization_strength=aae_class.ProximalAdagradOptimizer_l1_regularization_strength_generator,
            l2_regularization_strength=aae_class.ProximalAdagradOptimizer_l2_regularization_strength_generator
        )
    elif optimizer_name == "RMSPropOptimizer":
        return tf.train.RMSPropOptimizer(
            learning_rate=aae_class.learning_rate_generator, decay=aae_class.RMSPropOptimizer_decay_generator,
            momentum=aae_class.RMSPropOptimizer_momentum_generator, epsilon=aae_class.RMSPropOptimizer_epsilon_generator,
            centered=aae_class.RMSPropOptimizer_centered_generator)


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


def reshape_to_rgb_image(image_array, input_dim_x, input_dim_y):
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
        return DataLoading.read_svhn_from_mat('./data', one_hot=True)
    elif selected_dataset == "cifar10":
        return DataLoading.read_cifar10('./data', one_hot=True)
    elif selected_dataset == "custom":
        # TODO:
        print("not yet implemented")
        return


# TODO: maybe generate image grid unsupervised
# TODO: maybe generate image grid (semi-)supervised
# TODO: maybe generate image grid z_dim

