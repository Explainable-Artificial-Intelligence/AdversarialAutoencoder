import datetime
import os

import numpy as np
import tensorflow as tf

from util import DataLoading


def use_activation_function_for_layer(activation_function, layer):
    """
    uses the provided activation function for the layer and returns the new tensor
    :param activation_function: ["relu", "relu6", "crelu", "elu", "softplus", "softsign", "sigmoid", "tanh",
                                "leaky_relu", "linear"]
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
    elif activation_function == "leaky_relu":
        return tf.nn.leaky_relu(layer)
    elif activation_function == "linear":
        return layer


def get_tf_initializer(initializer_name, initializer_params):
    """
    wrapper function for the tensorflow initializers
    :param initializer_name: name of the initializer
    :param initializer_params: single value or tuple holding the parameter(s) of the initializer
    :return: instance of the initializer
    """

    if initializer_name == "constant_initializer":
        # value: A Python scalar, list or tuple of values, or a N-dimensional numpy array. All elements of the
        # initialized variable will be set to the corresponding value in the value argument.
        return tf.constant_initializer(value=initializer_params["value"])
    elif initializer_name == "random_normal_initializer":
        # mean: a python scalar or a scalar tensor. Mean of the random values to generate.
        # stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
        return tf.random_normal_initializer(mean=initializer_params["mean"], stddev=initializer_params["stddev"])
    elif initializer_name == "truncated_normal_initializer":
        # mean: a python scalar or a scalar tensor. Mean of the random values to generate.
        # stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
        return tf.truncated_normal_initializer(mean=initializer_params["mean"], stddev=initializer_params["stddev"])
    elif initializer_name == "random_uniform_initializer":
        # minval: A python scalar or a scalar tensor. Lower bound of the range of random values to generate.
        # maxval: A python scalar or a scalar tensor. Upper bound of the range of random values to generate.
        #         Defaults to 1 for float types.
        return tf.random_uniform_initializer(minval=initializer_params["minval"], maxval=initializer_params["maxval"])
    elif initializer_name == "uniform_unit_scaling_initializer":
        # factor: Float. A multiplicative factor by which the values will be scaled.
        return tf.uniform_unit_scaling_initializer(factor=initializer_params["factor"])
    elif initializer_name == "zeros_initializer":
        return tf.zeros_initializer()
    elif initializer_name == "ones_initializer":
        return tf.ones_initializer()
    elif initializer_name == "orthogonal_initializer":
        # gain: multiplicative factor to apply to the orthogonal matrix
        return tf.orthogonal_initializer(gain=initializer_params["gain"])


def get_decaying_learning_rate(decaying_learning_rate_name, decaying_learning_rate_params, global_step,
                               initial_learning_rate=0.1):
    """
    wrapper function for the decayed learning rate functions as defined in
    https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate
    :param decaying_learning_rate_name: ["exponential_decay", "inverse_time_decay", "natural_exp_decay",
    "piecewise_constant", "polynomial_decay"]
    :param decaying_learning_rate_params: dictionary holding the parameters for the decaying learning rate
    :param global_step: A scalar int32 or int64 Tensor or a Python number. Global step to use for the decay
    computation. Must not be negative.
    :param initial_learning_rate: A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
    :return:
    """

    if decaying_learning_rate_name == "exponential_decay":
        """
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        """
        decay_steps = decaying_learning_rate_params["decay_steps"]  # 100000
        decay_rate = decaying_learning_rate_params["decay_rate"]  # 0.96
        # If the argument staircase is True, then global_step / decay_steps is an integer division and the decayed
        # learning rate follows a staircase function.
        staircase = decaying_learning_rate_params["staircase"]  # False
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                                   decay_steps, decay_rate, staircase)

    elif decaying_learning_rate_name == "inverse_time_decay":
        """
        staircase=False:
            decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
        staircase=True:
            decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
        """
        decay_steps = decaying_learning_rate_params["decay_steps"]  # 1.0
        decay_rate = decaying_learning_rate_params["decay_rate"]  # 0.5
        staircase = decaying_learning_rate_params["staircase"]  # False
        learning_rate = tf.train.inverse_time_decay(initial_learning_rate, global_step,
                                                    decay_steps, decay_rate, staircase)

    elif decaying_learning_rate_name == "natural_exp_decay":
        """
        decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        """
        decay_steps = decaying_learning_rate_params["decay_steps"]  # 1.0
        decay_rate = decaying_learning_rate_params["decay_rate"]  # 0.5
        staircase = decaying_learning_rate_params["staircase"]  # False
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
        boundaries = decaying_learning_rate_params["boundaries"]  # [250]
        values = decaying_learning_rate_params["values"]  # [0.01, 0.001]
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
        decay_steps = decaying_learning_rate_params["decay_steps"]  # 10000
        end_learning_rate = decaying_learning_rate_params["end_learning_rate"]  # 0.00001
        power = decaying_learning_rate_params["power"]  # 1.0
        cycle = decaying_learning_rate_params["cycle"]  # False
        learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step, decay_steps,
                                                  end_learning_rate, power, cycle)

    elif decaying_learning_rate_name == "static":
        """
        Static learning rate.
        """
        learning_rate = decaying_learning_rate_params["learning_rate"]

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
        learning_rate = \
            get_decaying_learning_rate(decaying_learning_rate_name=decaying_learning_rate_name, global_step=global_step,
                                       initial_learning_rate=param_dict.get("learning_rate_" + sub_network_name),
                                       decaying_learning_rate_params=
                                       param_dict.get("decaying_learning_rate_params_" + sub_network_name))
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


def create_dense_layer(inputs, n_input_neurons, n_output_neurons, variable_scope_name, activation_function,
                       weight_initializer, weight_initializer_params, bias_initializer, bias_initializer_params,
                       drop_out_rate_input_layer, drop_out_rate_output_layer, is_training,
                       batch_normalization=None):
    """
    Used to create a dense layer.
    :param inputs: input tensor to the dense layer
    :param n_input_neurons: no. of input neurons
    :param n_output_neurons: no. of output neurons
    :param variable_scope_name: name of the entire dense layer
    :param activation_function: activation function to use. One of ["relu", "relu6", "crelu", "elu", "softplus",
    "softsign", "sigmoid", "tanh", "leaky_relu", "linear"]
    :param weight_initializer: which tf initializer should be used for initializing the weights
    :param weight_initializer_params: the parameters for initializing the weights
    :param bias_initializer: which tf initializer should be used for initializing the bias
    :param bias_initializer_params: the parameters for initializing the bias
    :param drop_out_rate_input_layer: The dropout rate of the input layer, between 0 and 1. E.g. "rate=0.1" would drop
    out 10% of input units.
    :param drop_out_rate_output_layer: The dropout rate of the output layer, between 0 and 1. E.g. "rate=0.1" would drop
    out 10% of output units.
    :param is_training: whether or not the the layer is in training mode.
    :param batch_normalization: one of ["pre_activation", "post_activation", None]: whether to use pre-, or
    post-activation batch normalization or no batch normalization at all
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(variable_scope_name, reuse=None):
        # apply drop out on the input layer
        inputs_drop_out = tf.layers.dropout(inputs, rate=drop_out_rate_input_layer)
        # create weights + biases
        weights = tf.get_variable("weights", shape=[n_input_neurons, n_output_neurons],
                                  initializer=get_tf_initializer(weight_initializer, weight_initializer_params))
        bias = tf.get_variable("bias", shape=[n_output_neurons],
                               initializer=get_tf_initializer(bias_initializer, bias_initializer_params))
        # create the layer
        output_layer = tf.add(tf.matmul(inputs_drop_out, weights), bias, name='matmul')

        # pre-activation batch normalization
        if batch_normalization == "pre_activation":
            # apply batch normalization
            output_layer_bn = tf.contrib.layers.batch_norm(output_layer, center=True, scale=True,
                                                           is_training=is_training)
            # apply the activation function
            output_layer_activ_fct = use_activation_function_for_layer(activation_function, output_layer_bn)
            # apply dropout
            output_layer_drop_out = tf.layers.dropout(output_layer_activ_fct, rate=drop_out_rate_output_layer)
            return output_layer_drop_out
        # post-activation batch normalization
        elif batch_normalization == "post_activation":
            # apply the activation function
            output_layer_activ_fct = use_activation_function_for_layer(activation_function,
                                                                                         output_layer)
            # apply batch normalization
            output_layer_bn = tf.contrib.layers.batch_norm(output_layer_activ_fct, center=True,
                                                           scale=True, is_training=is_training)
            # apply dropout
            output_layer_drop_out = tf.layers.dropout(output_layer_bn, rate=drop_out_rate_output_layer)
            return output_layer_drop_out
        # no batch normalization at all
        else:
            # apply the activation function
            output_layer_activ_fct = use_activation_function_for_layer(activation_function, output_layer)
            # apply dropout
            output_layer_drop_out = tf.layers.dropout(output_layer_activ_fct, rate=drop_out_rate_output_layer)
            return output_layer_drop_out


def create_convolutional_layer(inputs, variable_scope_name, filters, kernel_size, padding, strides, activation_function,
                               batch_normalization=None, is_training=True, name_prefix=None):
    """
    Used to create a convolutional layer
    :param inputs: 4D input tensor of shape [batch, in_height, in_width, in_channels]
    :param variable_scope_name: name_prefix of the conv. layer
    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size:  An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
    window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: One of "valid" or "same". Valid: no paddding at all; Same: p=ceil(k/2) with k=kernel size
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
    and width. Can be a single integer to specify the same value for all spatial dimensions.
    :param activation_function: activation function to use. One of ["relu", "relu6", "crelu", "elu", "softplus",
    "softsign", "sigmoid", "tanh", "leaky_relu", "linear"]
    :param batch_normalization: one of ["pre_activation", "post_activation", None]: whether to use pre-, or
    post-activation batch normalization or no batch normalization at all
    :param is_training: whether or not the the layer is in training mode.
    :param name_prefix: prefix for the name of the tensorflow variable
    :return: output tensor
    """

    if not name_prefix:
        name_prefix = ""
    else:
        name_prefix += "_"

    if isinstance(padding, int):
        padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]

    with tf.variable_scope(variable_scope_name, reuse=None):

        padded_inputs = tf.pad(inputs, padding)

        conv_layer = tf.layers.conv2d(inputs=padded_inputs, filters=filters, kernel_size=kernel_size, padding="valid",
                                      strides=strides, name=name_prefix + "conv_layer")

        # pre-activation batch normalization
        if batch_normalization == "pre_activation":
            conv_layer_bn = tf.contrib.layers.batch_norm(conv_layer, center=True, scale=True, is_training=is_training)
            return use_activation_function_for_layer(activation_function, conv_layer_bn)

        # post-activation batch normalization
        elif batch_normalization == "post_activation":
            conv_layer_activation_function_applied = use_activation_function_for_layer(activation_function, conv_layer)
            conv_layer_bn = tf.contrib.layers.batch_norm(conv_layer_activation_function_applied, center=True,
                                                         scale=True, is_training=is_training)
            return conv_layer_bn

        # no batch normalization at all
        else:
            return use_activation_function_for_layer(activation_function, conv_layer)


def create_transposed_convolutional_layer(inputs, variable_scope_name, filters, kernel_size, padding, strides,
                                          activation_function, batch_normalization=None, is_training=True,
                                          name_prefix=None):
    """
    Used to create a transposed convolutional layer
    :param inputs: 4D input tensor of shape [batch, in_height, in_width, in_channels]
    :param variable_scope_name: name_prefix of the conv. layer
    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size:  An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
    window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: `padding` is an integer tensor with shape `[n, 2]`, where n is the rank of `tensor`. For each
    dimension D of `input`, `paddings[D, 0]` indicates how many values to add before the contents of `tensor` in that
    dimension, and `paddings[D, 1]` indicates how many values to add after the contents of `tensor` in that dimension.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
    and width. Can be a single integer to specify the same value for all spatial dimensions.
    :param activation_function: activation function to use. One of ["relu", "relu6", "crelu", "elu", "softplus",
    "softsign", "sigmoid", "tanh", "leaky_relu", "linear"]
    :param batch_normalization: one of ["pre_activation", "post_activation", None]: whether to use pre-, or
    post-activation batch normalization or no batch normalization at all
    :param is_training: whether or not the the layer is in training mode.
    :param name_prefix: prefix for the name of the tensorflow variable
    :return: output tensor
    """

    if not name_prefix:
        name_prefix = ""

    if isinstance(padding, int):
        padding = [[padding, padding], [padding, padding], [padding, padding], [padding, padding]]

    with tf.variable_scope(variable_scope_name, reuse=None):

        padded_inputs = inputs

        if padding:
            padded_inputs = tf.pad(inputs, padding)

        conv_layer = tf.layers.conv2d_transpose(inputs=padded_inputs, filters=filters, kernel_size=kernel_size,
                                                padding="valid",
                                                strides=strides, name=name_prefix + "transposed_conv_layer")

        # pre-activation batch normalization
        if batch_normalization == "pre_activation":
            conv_layer_bn = tf.contrib.layers.batch_norm(conv_layer, center=True, scale=True, is_training=is_training)
            return use_activation_function_for_layer(activation_function, conv_layer_bn)

        # post-activation batch normalization
        elif batch_normalization == "post_activation":
            conv_layer_activation_function_applied = use_activation_function_for_layer(activation_function, conv_layer)
            conv_layer_bn = tf.contrib.layers.batch_norm(conv_layer_activation_function_applied, center=True,
                                                         scale=True, is_training=is_training)
            return conv_layer_bn

        # no batch normalization at all
        else:
            return use_activation_function_for_layer(activation_function, conv_layer)


def create_residual_block(inputs, variable_scope_name, filters, kernel_size,
                          use_transposed_conv_layers=False, activation_function="relu", strides=(1, 2)):
    """
    creates a residual block with the architecture as described in
    Learning Priors for Adversarial autoencoders available at: https://openreview.net/forum?id=rJSr0GZR-
        Architecture:
            Input feature map
            3 x 3 conv. out_channels RELU stride 2 pad 1
            3 x 3 conv. out_channels RELU stride 1 pad 1
            skip connection output = input + residual
            RELU
    :param inputs: 4D input tensor of shape [batch, in_height, in_width, in_channels]
    :param variable_scope_name: name of the residual block
    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
    window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: One of "valid" or "same". Valid: no paddding at all; Same: p=ceil(k/2) with k=kernel size
    :param use_transposed_conv_layers: whether to use transposed convolutional layers or not
    :param activation_function: activation function for the output to use. One of ["relu", "relu6", "crelu", "elu",
    "softplus", "softsign", "sigmoid", "tanh", "leaky_relu", "linear"]
    :param strides: tuple of integers holding the strides to use for the first and second conv layer
    :return: output tensor
    """
    with tf.variable_scope(variable_scope_name, reuse=None):
        if use_transposed_conv_layers:

            first_layer = create_transposed_convolutional_layer(inputs, variable_scope_name, filters=filters,
                                                                kernel_size=kernel_size, padding=None,
                                                                strides=strides[0], activation_function="relu",
                                                                name_prefix="layer_1")

            second_layer = create_transposed_convolutional_layer(first_layer, variable_scope_name,
                                                                 filters=filters,
                                                                 kernel_size=kernel_size, padding=None,
                                                                 strides=strides[1], activation_function="relu",
                                                                 name_prefix="layer_2")
        else:

            first_layer = create_convolutional_layer(inputs, variable_scope_name, filters=filters,
                                                     kernel_size=kernel_size, padding=1, strides=strides[0],
                                                     activation_function="relu", name_prefix="layer_1")

            second_layer = create_convolutional_layer(first_layer, variable_scope_name, filters=filters,
                                                      kernel_size=kernel_size, padding=1, strides=strides[1],
                                                      activation_function="relu", name_prefix="layer_2")
        # skip connection output = input + residual
        # output = inputs + second_layer
        # TODO: include skip connection
        output = second_layer
        # output = first_layer

        # apply activation function to the output
        return use_activation_function_for_layer(activation_function, output)


def create_pooling_layer(inputs, pool_size, strides, pooling_to_use, variable_scope_name):
    """
    Creates either a max or a average pooling layer for the inputs.
    :param inputs: The tensor over which to pool. Must have rank 4.
    :param pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the
    pooling window. Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation. Can be a
    single integer to specify the same value for all spatial dimensions.
    :param pooling_to_use: The pooling to use: One of "max", "avg"
    :param variable_scope_name: Name of the pooling layer
    :return: output tensor
    """
    with tf.variable_scope(variable_scope_name, reuse=None):
        if pooling_to_use == "max":
            return tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides)
        elif pooling_to_use == "avg":
            return tf.layers.average_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides)


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


def get_min_and_max_per_dim_on_latent_space(z_dim):

    # TODO: no more hard coded
    data = np.load('../data/dimension_reduced_data/SVHN/pca/SVHN_pca_z_dim_8' + '.npy')

    latent_space_min_max_per_dim = []

    print(data.shape)

    for dim in range(z_dim):
        latent_space_min_max_per_dim.append({"min": np.amin(data[:, dim]), "max": np.amax(data[:, dim])})

    return latent_space_min_max_per_dim


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
    if selected_autoencoder in ["SemiSupervisedAdversarialAutoencoder", "UnsupervisedClusteringAdversarialAutoencoder",
                                "DimensionalityReductionAdversarialAutoencoder"]:
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
    :param aae_class: AAE instance to access some important fields from it
    :param function_params: tuple of the variable denoting whether to get the bias or the weights and the layer name
    :return: the bias/weights for the specific layer
    """

    bias_or_weights, layer_name = function_params
    all_values = None
    if bias_or_weights == "bias":
        all_values = get_all_layer_vars(aae_class, "bias")
    elif bias_or_weights == "weights":
        all_values = get_all_layer_vars(aae_class, "weights")

    return all_values[layer_name]

