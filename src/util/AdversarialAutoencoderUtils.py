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

    # TODO: params of different learning rates as kwargs

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


def get_input_data(selected_dataset, color_scale, data_normalized=False):
    """
    returns the input data set based on self.selected_dataset
    :return: object holding the train data, the test data and the validation data
    """
    return DataLoading.get_input_data(selected_dataset, color_scale=color_scale, data_normalized=data_normalized)


def get_min_and_max_per_dim_on_latent_space(z_dim):

    # TODO: no more hard coded
    data = np.load('../data/dimension_reduced_data/SVHN/pca/SVHN_pca_z_dim_8' + '.npy')

    latent_space_min_max_per_dim = []

    print(data.shape)

    for dim in range(z_dim):
        latent_space_min_max_per_dim.append({"min": np.amin(data[:, dim]), "max": np.amax(data[:, dim])})

    return latent_space_min_max_per_dim


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
        mu = np.mean(latent_representations_current_epoch, axis=0)

        pca = PCA(n_components=2)
        pca.fit(latent_representations_current_epoch)
        latent_representations_current_epoch = pca.transform(latent_representations_current_epoch)

        Xhat = np.dot(latent_representations_current_epoch[:, :2], pca.components_[:2, :])
        Xhat += mu

        print()
        print(Xhat.shape)
        print(Xhat[0])

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
    :param discriminator_pos: output of the discriminator for the positive samples p(z) (from real data distribution)
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
    # if the aae is not trained enough/has unfavorable parameters, it's possible that the reconstruction can hold
    # pixels with negative value
    # if (created_img < 0).any():
        # created_img = (created_img - np.min(created_img)) / (np.max(created_img) - np.min(created_img))
        # created_img = created_img / np.linalg.norm(created_img)
        # created_img = np.abs(created_img)
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
    plt.subplot(4, 3, 8)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
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
    :param batch_labels: labels of the current batch
    :param discriminator_gaussian_neg: output of the gaussian discriminator for the negative samples q(z) (generated by
    the generator)
    :param discriminator_gaussian_pos: output of the gaussian discriminator  for the positive samples p(z) (from real
    data distribution)
    :param discriminator_cat_neg: output of the categorical discriminator for the negative samples
    :param discriminator_cat_pos: output of the categorical discriminator for the positive samples
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
    # if the aae is not trained enough/has unfavorable parameters, it's possible that the reconstruction can hold
    # pixels with negative value
    if (created_img < 0).any():
        created_img = np.abs(created_img)
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
    plt.subplot(5, 4, 12)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
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


def visualize_cluster_heads(aae_class, cluster_heads, epoch, mini_batch_i):

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=6, ncols=3)     # TODO: depending on n_clusters

    for i, cluster_head in enumerate(cluster_heads):
        plt.subplot(6, 3, i+1)

        img = reshape_image_array(aae_class, cluster_head)
        if aae_class.color_scale == "gray_scale":
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title("cluster head: " + str(i))

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '_cluster_heads.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_minibatch_summary_image_unsupervised_clustering(aae_class, real_dist, latent_representation, batch_x, decoder_output,
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
    :param batch_labels: labels of the current batch
    :param discriminator_gaussian_neg: output of the gaussian discriminator for the negative samples q(z) (generated by
    the generator)
    :param discriminator_gaussian_pos: output of the gaussian discriminator  for the positive samples p(z) (from real
    data distribution)
    :param discriminator_cat_neg: output of the categorical discriminator for the negative samples
    :param discriminator_cat_pos: output of the categorical discriminator for the positive samples
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
                                        aae_class.performance_over_time["generator_losses"])]

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
                                                          performance_dict["generator_losses"])]
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
    # if the aae is not trained enough/has unfavorable parameters, it's possible that the reconstruction can hold
    # pixels with negative value
    if (created_img < 0).any():
        created_img = np.abs(created_img)
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

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
    plt.subplot(5, 4, 12)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
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

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_reconstruction_grid(aae_class, real_images, reconstructed_images, epoch, batch_number):

    # TODO: depending on batch_size
    nx, ny = 10, 10

    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    # counter for the two arrays holding the real and the reconstruction images
    image_array_counter = 0

    # iterate over the image grid
    for i, g in enumerate(gs):

        ax = plt.subplot(g)

        if i % 2 == 0:
            # plot one input image..
            real_img = real_images[image_array_counter, :]
            img = reshape_image_array(aae_class, real_img)
            if aae_class.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
        else:
            # .. and its reconstruction
            created_img = reconstructed_images[image_array_counter, :]
            img = reshape_image_array(aae_class, created_img)
            if aae_class.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            # we plotted one input image and its reconstruction, so we go on to the next images
            image_array_counter += 1

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/'
                + str(epoch) + "_" + str(batch_number) + "_reconstruction_grid" + '.png')

    plt.close("all")


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
    if selected_autoencoder == "SemiSupervisedAdversarialAutoencoder" \
            or selected_autoencoder == "UnsupervisedClusteringAdversarialAutoencoder":
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

    # TODO: experimental!
    if False:
        input_layer_weights = all_weights["encoder_dense_layer_1"]
        visualize_weights_single_layer_as_img_grid(input_layer_weights, aae_class.input_dim, "encoder_dense_layer_1",
                                                   "weights", result_path_weight_img, epoch)


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


def visualize_weights_single_layer_as_img_grid(weights, input_dim, layer_name, var_to_visualize, result_path, epoch):

    # TODO: color scale!
    #

    # we assume quadratic input images
    nx = int(np.sqrt(input_dim))

    # increase figure size
    plt.rcParams["figure.figsize"] = (30, 30)

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

    plt.title("Epoch: " + str(epoch))
    plt.savefig(result_path + layer_name + "_" + var_to_visualize + "image_grid.png")

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

# TODO: maybe generate image grid unsupervised
# TODO: maybe generate image grid (semi-)supervised
# TODO: maybe generate image grid z_dim

