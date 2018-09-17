import copy
import random
import itertools
import numpy as np
from util.Distributions import draw_from_np_distribution


def get_result_path_for_selected_autoencoder(selected_autoencoder):
    """
    returns the result path based on the selected autoencoder
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :return: string, where to save the result files for training the autoencoder
    """
    if selected_autoencoder == "Unsupervised":
        return '../../results/Unsupervised'
    elif selected_autoencoder == "Supervised":
        return '../../results/Supervised'
    elif selected_autoencoder == "SemiSupervised":
        return '../../results/SemiSupervised'
    elif selected_autoencoder == "UnsupervisedClustering":
        return '../../results/UnsupervisedClustering'
    elif selected_autoencoder == "DimensionalityReduction":
        return '../../results/DimensionalityReduction'
    elif selected_autoencoder == "IncorporatingLabelInformation":
        return '../../results/IncorporatingLabelInformation'
    else:
        print(selected_autoencoder + " has no result path associated with it!")
        raise NotImplementedError


def get_default_parameters_mnist():
    """
    returns the default parameters for the MNIST dataset
    :return: dictionary holding the parameters needed to create the Autoencoder
    """
    return {'batch_size': 100, 'n_epochs': 10, 'input_dim_x': 28, 'input_dim_y': 28, 'z_dim': 8, 'n_classes': 10,
            'color_scale': "gray_scale", 'verbose': True, 'save_final_model': False, 'write_tensorboard': False,
            'n_labeled': 1000,  # for semi-supervised
            'selected_dataset': "MNIST",
            'summary_image_frequency': 5,  # create a summary image of the learning process every 5 epochs
            'n_neurons_of_hidden_layer_x_autoencoder': [1000, 1000],  # 1000, 500, 250, 125
            'n_neurons_of_hidden_layer_x_discriminator': [1000, 1000],  # 500, 250, 125
            'n_neurons_of_hidden_layer_x_discriminator_c': [1000, 1000],  # for semi-supervised
            'n_neurons_of_hidden_layer_x_discriminator_g': [1000, 1000],  # for semi-supervised

            'dropout_encoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_decoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator': [0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator_c': [0.0, 0.0, 0.0],
            'dropout_discriminator_g': [0.0, 0.0, 0.0],

            'batch_normalization_encoder': [None, None, None, None, None],
            'batch_normalization_decoder': [None, None, None, None, None],
            'batch_normalization_discriminator': [None, None, None, None, None],
            # for semi-supervised:
            'batch_normalization_discriminator_c': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_discriminator_g': ["post_activation", "post_activation", "post_activation"],

            'bias_initializer_encoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_decoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_c': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_g': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],

            'bias_initializer_params_encoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_decoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_discriminator': [{}, {}, {}, {}],
            'bias_initializer_params_discriminator_c': [{}, {}, {}],
            'bias_initializer_params_discriminator_g': [{}, {}, {}],

            'weights_initializer_encoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_decoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_discriminator': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                  "truncated_normal_initializer"],
            'weights_initializer_discriminator_c': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],
            'weights_initializer_discriminator_g': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],

            'weights_initializer_params_encoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_decoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                         {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_c': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_g': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],

            'activation_function_encoder': ['relu', 'relu', 'linear'],
            'activation_function_decoder': ['relu', 'relu', 'sigmoid'],
            'activation_function_discriminator': ['relu', 'relu', 'linear'],
            'activation_function_discriminator_c': ['relu', 'relu', 'linear'],  # for semi-supervised
            'activation_function_discriminator_g': ['relu', 'relu', 'linear'],  # for semi-supervised

            'decaying_learning_rate_name_autoencoder': "static",
            'decaying_learning_rate_name_discriminator': "static",
            'decaying_learning_rate_name_generator': "static",
            'decaying_learning_rate_name_discriminator_gaussian': "static",
            'decaying_learning_rate_name_discriminator_categorical': "static",
            'decaying_learning_rate_name_supervised_encoder': "static",

            'decaying_learning_rate_params_autoencoder': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_generator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_gaussian': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_categorical': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_supervised_encoder': {"learning_rate": 0.0001},

            'optimizer_autoencoder': 'AdamOptimizer',
            'optimizer_discriminator': 'AdamOptimizer',
            'optimizer_generator': 'AdamOptimizer',
            'optimizer_discriminator_gaussian': 'AdamOptimizer',
            'optimizer_discriminator_categorical': 'AdamOptimizer',
            'optimizer_supervised_encoder': 'AdamOptimizer',
            'AdadeltaOptimizer_rho_autoencoder': 0.95, 'AdadeltaOptimizer_epsilon_autoencoder': 1e-08,
            'AdadeltaOptimizer_rho_discriminator': 0.95, 'AdadeltaOptimizer_epsilon_discriminator': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_gaussian': 0.95,
            'AdadeltaOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_categorical': 0.95, 'AdadeltaOptimizer_epsilon_categorical': 1e-08,
            'AdadeltaOptimizer_rho_supervised_encoder': 0.95, 'AdadeltaOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdadeltaOptimizer_rho_generator': 0.95, 'AdadeltaOptimizer_epsilon_generator': 1e-08,
            'AdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'AdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'MomentumOptimizer_momentum_autoencoder': 0.9, 'MomentumOptimizer_use_nesterov_autoencoder': False,
            'MomentumOptimizer_momentum_discriminator': 0.9, 'MomentumOptimizer_use_nesterov_discriminator': False,
            'MomentumOptimizer_momentum_discriminator_gaussian': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_gaussian': False,
            'MomentumOptimizer_momentum_discriminator_categorical': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_categorical': False,
            'MomentumOptimizer_momentum_supervised_encoder': 0.9,
            'MomentumOptimizer_use_nesterov_supervised_encoder': False,
            'MomentumOptimizer_momentum_generator': 0.9, 'MomentumOptimizer_use_nesterov_generator': False,
            'AdamOptimizer_beta1_autoencoder': 0.9, 'AdamOptimizer_beta2_autoencoder': 0.999,
            'AdamOptimizer_epsilon_autoencoder': 1e-08, 'AdamOptimizer_beta1_discriminator': 0.9,
            'AdamOptimizer_beta2_discriminator': 0.999, 'AdamOptimizer_epsilon_discriminator': 1e-08,
            'AdamOptimizer_beta1_discriminator_gaussian': 0.9,
            'AdamOptimizer_beta2_discriminator_gaussian': 0.999,
            'AdamOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdamOptimizer_beta1_discriminator_categorical': 0.9,
            'AdamOptimizer_beta2_discriminator_categorical': 0.999,
            'AdamOptimizer_epsilon_discriminator_categorical': 1e-08,
            'AdamOptimizer_beta1_supervised_encoder': 0.9,
            'AdamOptimizer_beta2_supervised_encoder': 0.999, 'AdamOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdamOptimizer_beta1_generator': 0.9, 'AdamOptimizer_beta2_generator': 0.999,
            'AdamOptimizer_epsilon_generator': 1e-08, 'FtrlOptimizer_learning_rate_power_autoencoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_gaussian': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_categorical': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_learning_rate_power_supervised_encoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_learning_rate_power_generator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_generator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_generator': 0.0,
            'RMSPropOptimizer_decay_autoencoder': 0.9,
            'RMSPropOptimizer_momentum_autoencoder': 0.0,
            'RMSPropOptimizer_epsilon_autoencoder': 1e-10, 'RMSPropOptimizer_centered_autoencoder': False,
            'RMSPropOptimizer_decay_discriminator': 0.9, 'RMSPropOptimizer_momentum_discriminator': 0.0,
            'RMSPropOptimizer_epsilon_discriminator': 1e-10,
            'RMSPropOptimizer_centered_discriminator': False,
            'RMSPropOptimizer_decay_discriminator_gaussian': 0.9,
            'RMSPropOptimizer_momentum_discriminator_gaussian': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_gaussian': 1e-10,
            'RMSPropOptimizer_centered_discriminator_gaussian': False,
            'RMSPropOptimizer_decay_discriminator_categorical': 0.9,
            'RMSPropOptimizer_momentum_discriminator_categorical': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_categorical': 1e-10,
            'RMSPropOptimizer_centered_discriminator_categorical': False,
            'RMSPropOptimizer_decay_supervised_encoder': 0.9, 'RMSPropOptimizer_momentum_supervised_encoder': 0.0,
            'RMSPropOptimizer_epsilon_supervised_encoder': 1e-10,
            'RMSPropOptimizer_centered_supervised_encoder': False,
            'RMSPropOptimizer_decay_generator': 0.9, 'RMSPropOptimizer_momentum_generator': 0.0,
            'RMSPropOptimizer_epsilon_generator': 1e-10, 'RMSPropOptimizer_centered_generator': False,
            'loss_function_discriminator': 'sigmoid_cross_entropy',
            'loss_function_discriminator_gaussian': 'sigmoid_cross_entropy',
            'loss_function_discriminator_categorical': 'sigmoid_cross_entropy',
            'loss_function_generator': 'sigmoid_cross_entropy'}


def get_default_parameters_svhn():
    """
    returns the default parameters for the MNIST dataset
    :return: dictionary holding the parameters needed to create the Autoencoder
    """
    return {'batch_size': 100, 'n_epochs': 10, 'input_dim_x': 32, 'input_dim_y': 32, 'z_dim': 2, 'n_classes': 10,
            'color_scale': "rgb_scale", 'verbose': True, 'save_final_model': False, 'write_tensorboard': False,
            'n_labeled': 1000,  # for semi-supervised
            'selected_dataset': "SVHN",
            'summary_image_frequency': 5,  # create a summary image of the learning process every 5 epochs
            'n_neurons_of_hidden_layer_x_autoencoder': [3000, 1500, 750, 375],
            'n_neurons_of_hidden_layer_x_discriminator': [3000, 1500, 750, 375],
            'n_neurons_of_hidden_layer_x_discriminator_c': [1000, 1000],  # for semi-supervised
            'n_neurons_of_hidden_layer_x_discriminator_g': [1000, 1000],  # for semi-supervised

            'dropout_encoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_decoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator_c': [0.0, 0.0, 0.0],
            'dropout_discriminator_g': [0.0, 0.0, 0.0],

            'batch_normalization_encoder': ["post_activation", "post_activation", "post_activation", "post_activation",
                                            "post_activation"],
            'batch_normalization_decoder': ["post_activation", "post_activation", "post_activation", "post_activation",
                                            "post_activation"],
            'batch_normalization_discriminator': [None, None, None, None, None],
            # for semi-supervised:
            'batch_normalization_discriminator_c': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_discriminator_g': ["post_activation", "post_activation", "post_activation"],

            'bias_initializer_encoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                         "zeros_initializer", "zeros_initializer"],
            'bias_initializer_decoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                         "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                               "zeros_initializer"],
            'bias_initializer_discriminator_c': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_g': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],

            'bias_initializer_params_encoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_decoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_discriminator': [{}, {}, {}, {}],
            'bias_initializer_params_discriminator_c': [{}, {}, {}],
            'bias_initializer_params_discriminator_g': [{}, {}, {}],

            'weights_initializer_encoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_decoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_discriminator': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                  "truncated_normal_initializer", "truncated_normal_initializer"],
            'weights_initializer_discriminator_c': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],
            'weights_initializer_discriminator_g': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],

            'weights_initializer_params_encoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_decoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                         {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_c': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_g': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],

            'activation_function_encoder': ['relu', 'relu', 'relu', 'relu', 'linear'],
            'activation_function_decoder': ['sigmoid', 'relu', 'relu', 'relu', 'sigmoid'],
            'activation_function_discriminator': ['relu', 'relu', 'relu', 'relu', 'linear'],
            'activation_function_discriminator_c': ['relu', 'relu', 'linear'],  # for semi-supervised
            'activation_function_discriminator_g': ['relu', 'relu', 'linear'],  # for semi-supervised

            'decaying_learning_rate_name_autoencoder': "piecewise_constant",
            'decaying_learning_rate_name_discriminator': "piecewise_constant",
            'decaying_learning_rate_name_generator': "piecewise_constant",
            'decaying_learning_rate_name_discriminator_gaussian': "static",
            'decaying_learning_rate_name_discriminator_categorical': "static",
            'decaying_learning_rate_name_supervised_encoder': "static",

            'decaying_learning_rate_params_autoencoder': {"boundaries": [250], "values": [0.0001, 0.00001]},
            'decaying_learning_rate_params_discriminator': {"boundaries": [250], "values": [0.01, 0.001]},
            'decaying_learning_rate_params_generator': {"boundaries": [250], "values": [0.01, 0.001]},
            'decaying_learning_rate_params_discriminator_gaussian': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_categorical': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_supervised_encoder': {"learning_rate": 0.0001},


            'optimizer_autoencoder': 'AdamOptimizer',
            'optimizer_discriminator': 'AdamOptimizer',
            'optimizer_generator': 'AdamOptimizer',
            'optimizer_discriminator_gaussian': 'AdamOptimizer',
            'optimizer_discriminator_categorical': 'AdamOptimizer',
            'optimizer_supervised_encoder': 'AdamOptimizer',
            'AdadeltaOptimizer_rho_autoencoder': 0.95, 'AdadeltaOptimizer_epsilon_autoencoder': 1e-08,
            'AdadeltaOptimizer_rho_discriminator': 0.95, 'AdadeltaOptimizer_epsilon_discriminator': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_gaussian': 0.95,
            'AdadeltaOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_categorical': 0.95, 'AdadeltaOptimizer_epsilon_categorical': 1e-08,
            'AdadeltaOptimizer_rho_supervised_encoder': 0.95, 'AdadeltaOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdadeltaOptimizer_rho_generator': 0.95, 'AdadeltaOptimizer_epsilon_generator': 1e-08,
            'AdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'AdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'MomentumOptimizer_momentum_autoencoder': 0.9, 'MomentumOptimizer_use_nesterov_autoencoder': False,
            'MomentumOptimizer_momentum_discriminator': 0.9, 'MomentumOptimizer_use_nesterov_discriminator': False,
            'MomentumOptimizer_momentum_discriminator_gaussian': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_gaussian': False,
            'MomentumOptimizer_momentum_discriminator_categorical': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_categorical': False,
            'MomentumOptimizer_momentum_supervised_encoder': 0.9,
            'MomentumOptimizer_use_nesterov_supervised_encoder': False,
            'MomentumOptimizer_momentum_generator': 0.9, 'MomentumOptimizer_use_nesterov_generator': False,
            'AdamOptimizer_beta1_autoencoder': 0.9, 'AdamOptimizer_beta2_autoencoder': 0.999,
            'AdamOptimizer_epsilon_autoencoder': 1e-08, 'AdamOptimizer_beta1_discriminator': 0.9,
            'AdamOptimizer_beta2_discriminator': 0.999, 'AdamOptimizer_epsilon_discriminator': 1e-08,
            'AdamOptimizer_beta1_discriminator_gaussian': 0.9,
            'AdamOptimizer_beta2_discriminator_gaussian': 0.999,
            'AdamOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdamOptimizer_beta1_discriminator_categorical': 0.9,
            'AdamOptimizer_beta2_discriminator_categorical': 0.999,
            'AdamOptimizer_epsilon_discriminator_categorical': 1e-08,
            'AdamOptimizer_beta1_supervised_encoder': 0.9,
            'AdamOptimizer_beta2_supervised_encoder': 0.999, 'AdamOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdamOptimizer_beta1_generator': 0.9, 'AdamOptimizer_beta2_generator': 0.999,
            'AdamOptimizer_epsilon_generator': 1e-08, 'FtrlOptimizer_learning_rate_power_autoencoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_gaussian': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_categorical': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_learning_rate_power_supervised_encoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_learning_rate_power_generator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_generator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_generator': 0.0,
            'RMSPropOptimizer_decay_autoencoder': 0.9,
            'RMSPropOptimizer_momentum_autoencoder': 0.0,
            'RMSPropOptimizer_epsilon_autoencoder': 1e-10, 'RMSPropOptimizer_centered_autoencoder': False,
            'RMSPropOptimizer_decay_discriminator': 0.9, 'RMSPropOptimizer_momentum_discriminator': 0.0,
            'RMSPropOptimizer_epsilon_discriminator': 1e-10,
            'RMSPropOptimizer_centered_discriminator': False,
            'RMSPropOptimizer_decay_discriminator_gaussian': 0.9,
            'RMSPropOptimizer_momentum_discriminator_gaussian': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_gaussian': 1e-10,
            'RMSPropOptimizer_centered_discriminator_gaussian': False,
            'RMSPropOptimizer_decay_discriminator_categorical': 0.9,
            'RMSPropOptimizer_momentum_discriminator_categorical': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_categorical': 1e-10,
            'RMSPropOptimizer_centered_discriminator_categorical': False,
            'RMSPropOptimizer_decay_supervised_encoder': 0.9, 'RMSPropOptimizer_momentum_supervised_encoder': 0.0,
            'RMSPropOptimizer_epsilon_supervised_encoder': 1e-10,
            'RMSPropOptimizer_centered_supervised_encoder': False,
            'RMSPropOptimizer_decay_generator': 0.9, 'RMSPropOptimizer_momentum_generator': 0.0,
            'RMSPropOptimizer_epsilon_generator': 1e-10, 'RMSPropOptimizer_centered_generator': False,
            'loss_function_discriminator': 'sigmoid_cross_entropy',
            'loss_function_discriminator_gaussian': 'sigmoid_cross_entropy',
            'loss_function_discriminator_categorical': 'sigmoid_cross_entropy',
            'loss_function_generator': 'sigmoid_cross_entropy'}


def get_default_parameters_cifar10():
    """
    returns the default parameters for the MNIST dataset
    :return: dictionary holding the parameters needed to create the Autoencoder
    """
    return {'batch_size': 100, 'n_epochs': 10, 'input_dim_x': 32, 'input_dim_y': 32, 'z_dim': 2, 'n_classes': 10,
            'color_scale': "rgb_scale", 'verbose': True, 'save_final_model': False, 'write_tensorboard': False,
            'n_labeled': 1000,  # for semi-supervised,
            'selected_dataset': "cifar10",
            'summary_image_frequency': 5,  # create a summary image of the learning process every 5 epochs
            'n_neurons_of_hidden_layer_x_autoencoder': [1000, 1000],
            'n_neurons_of_hidden_layer_x_discriminator': [1000, 1000],
            'n_neurons_of_hidden_layer_x_discriminator_c': [1000, 1000],  # for semi-supervised
            'n_neurons_of_hidden_layer_x_discriminator_g': [1000, 1000],  # for semi-supervised

            'dropout_encoder': [0.0, 0.0, 0.0],
            'dropout_decoder': [0.0, 0.0, 0.0],
            'dropout_discriminator': [0.0, 0.0, 0.0],
            'dropout_discriminator_c': [0.0, 0.0, 0.0],
            'dropout_discriminator_g': [0.0, 0.0, 0.0],

            'batch_normalization_encoder': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_decoder': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_discriminator': [None, None, None],
            # for semi-supervised:
            'batch_normalization_discriminator_c': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_discriminator_g': ["post_activation", "post_activation", "post_activation"],

            'bias_initializer_encoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_decoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_c': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_g': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],

            'bias_initializer_params_encoder': [{}, {}, {}],
            'bias_initializer_params_decoder': [{}, {}, {}],
            'bias_initializer_params_discriminator': [{}, {}, {}],
            'bias_initializer_params_discriminator_c': [{}, {}, {}],
            'bias_initializer_params_discriminator_g': [{}, {}, {}],

            'weights_initializer_encoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_decoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_discriminator': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                  "truncated_normal_initializer"],
            'weights_initializer_discriminator_c': ["truncated_normal_initializer", "truncated_normal_initializer"],
            'weights_initializer_discriminator_g': ["truncated_normal_initializer", "truncated_normal_initializer"],

            'weights_initializer_params_encoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_decoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                         {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_c': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_g': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],

            'activation_function_encoder': ['relu', 'relu', 'linear'],
            'activation_function_decoder': ['sigmoid', 'relu', 'sigmoid'],
            'activation_function_discriminator': ['relu', 'relu', 'linear'],
            'activation_function_discriminator_c': ['relu', 'relu', 'linear'],  # for semi-supervised
            'activation_function_discriminator_g': ['relu', 'relu', 'linear'],  # for semi-supervised

            'decaying_learning_rate_name_autoencoder': "static",
            'decaying_learning_rate_name_discriminator': "static",
            'decaying_learning_rate_name_generator': "static",
            'decaying_learning_rate_name_discriminator_gaussian': "static",
            'decaying_learning_rate_name_discriminator_categorical': "static",
            'decaying_learning_rate_name_supervised_encoder': "static",

            'decaying_learning_rate_params_autoencoder': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_generator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_gaussian': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_categorical': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_supervised_encoder': {"learning_rate": 0.0001},

            'optimizer_autoencoder': 'AdamOptimizer',
            'optimizer_discriminator': 'AdamOptimizer',
            'optimizer_generator': 'AdamOptimizer',
            'optimizer_discriminator_gaussian': 'AdamOptimizer',
            'optimizer_discriminator_categorical': 'AdamOptimizer',
            'optimizer_supervised_encoder': 'AdamOptimizer',
            'AdadeltaOptimizer_rho_autoencoder': 0.95, 'AdadeltaOptimizer_epsilon_autoencoder': 1e-08,
            'AdadeltaOptimizer_rho_discriminator': 0.95, 'AdadeltaOptimizer_epsilon_discriminator': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_gaussian': 0.95,
            'AdadeltaOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_categorical': 0.95, 'AdadeltaOptimizer_epsilon_categorical': 1e-08,
            'AdadeltaOptimizer_rho_supervised_encoder': 0.95, 'AdadeltaOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdadeltaOptimizer_rho_generator': 0.95, 'AdadeltaOptimizer_epsilon_generator': 1e-08,
            'AdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'AdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'MomentumOptimizer_momentum_autoencoder': 0.9, 'MomentumOptimizer_use_nesterov_autoencoder': False,
            'MomentumOptimizer_momentum_discriminator': 0.9, 'MomentumOptimizer_use_nesterov_discriminator': False,
            'MomentumOptimizer_momentum_discriminator_gaussian': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_gaussian': False,
            'MomentumOptimizer_momentum_discriminator_categorical': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_categorical': False,
            'MomentumOptimizer_momentum_supervised_encoder': 0.9,
            'MomentumOptimizer_use_nesterov_supervised_encoder': False,
            'MomentumOptimizer_momentum_generator': 0.9, 'MomentumOptimizer_use_nesterov_generator': False,
            'AdamOptimizer_beta1_autoencoder': 0.9, 'AdamOptimizer_beta2_autoencoder': 0.999,
            'AdamOptimizer_epsilon_autoencoder': 1e-08, 'AdamOptimizer_beta1_discriminator': 0.9,
            'AdamOptimizer_beta2_discriminator': 0.999, 'AdamOptimizer_epsilon_discriminator': 1e-08,
            'AdamOptimizer_beta1_discriminator_gaussian': 0.9,
            'AdamOptimizer_beta2_discriminator_gaussian': 0.999,
            'AdamOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdamOptimizer_beta1_discriminator_categorical': 0.9,
            'AdamOptimizer_beta2_discriminator_categorical': 0.999,
            'AdamOptimizer_epsilon_discriminator_categorical': 1e-08,
            'AdamOptimizer_beta1_supervised_encoder': 0.9,
            'AdamOptimizer_beta2_supervised_encoder': 0.999, 'AdamOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdamOptimizer_beta1_generator': 0.9, 'AdamOptimizer_beta2_generator': 0.999,
            'AdamOptimizer_epsilon_generator': 1e-08, 'FtrlOptimizer_learning_rate_power_autoencoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_gaussian': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_categorical': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_learning_rate_power_supervised_encoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_learning_rate_power_generator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_generator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_generator': 0.0,
            'RMSPropOptimizer_decay_autoencoder': 0.9,
            'RMSPropOptimizer_momentum_autoencoder': 0.0,
            'RMSPropOptimizer_epsilon_autoencoder': 1e-10, 'RMSPropOptimizer_centered_autoencoder': False,
            'RMSPropOptimizer_decay_discriminator': 0.9, 'RMSPropOptimizer_momentum_discriminator': 0.0,
            'RMSPropOptimizer_epsilon_discriminator': 1e-10,
            'RMSPropOptimizer_centered_discriminator': False,
            'RMSPropOptimizer_decay_discriminator_gaussian': 0.9,
            'RMSPropOptimizer_momentum_discriminator_gaussian': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_gaussian': 1e-10,
            'RMSPropOptimizer_centered_discriminator_gaussian': False,
            'RMSPropOptimizer_decay_discriminator_categorical': 0.9,
            'RMSPropOptimizer_momentum_discriminator_categorical': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_categorical': 1e-10,
            'RMSPropOptimizer_centered_discriminator_categorical': False,
            'RMSPropOptimizer_decay_supervised_encoder': 0.9, 'RMSPropOptimizer_momentum_supervised_encoder': 0.0,
            'RMSPropOptimizer_epsilon_supervised_encoder': 1e-10,
            'RMSPropOptimizer_centered_supervised_encoder': False,
            'RMSPropOptimizer_decay_generator': 0.9, 'RMSPropOptimizer_momentum_generator': 0.0,
            'RMSPropOptimizer_epsilon_generator': 1e-10, 'RMSPropOptimizer_centered_generator': False,
            'loss_function_discriminator': 'sigmoid_cross_entropy',
            'loss_function_discriminator_gaussian': 'sigmoid_cross_entropy',
            'loss_function_discriminator_categorical': 'sigmoid_cross_entropy',
            'loss_function_generator': 'hinge_loss'}


def get_default_parameters_mass_spec():
    """
    returns the default parameters for the MNIST dataset
    :return: dictionary holding the parameters needed to create the Autoencoder
    """
    return {'batch_size': 100, 'n_epochs': 10, 'input_dim_x': 1, 'input_dim_y': 150, 'z_dim': 2, 'n_classes': 2,
            'color_scale': "gray_scale", 'verbose': True, 'save_final_model': False, 'write_tensorboard': False,
            'n_labeled': 1000,  'only_train_autoencoder': True, 'selected_dataset': "mass_spec",
            'summary_image_frequency': 5,  # create a summary image of the learning process every 5 epochs

            'mz_loss_factor': 1,
            'intensity_loss_factor': 1,

            'mass_spec_data_properties': {"organism_name": "yeast", "peak_encoding": "only_intensities",
                                          "use_smoothed_intensities": True, "data_subset": None,
                                          "n_peaks_to_keep": 50, "max_intensity_value": 2000,
                                          "max_mz_value": 2000, "charge": None, "normalize_data": False,
                                          "include_molecular_weight_in_encoding": False,
                                          "include_charge_in_encoding": False,
                                          "smoothness_params": {"smoothing_method": "loess",
                                                                "smoothness_frac": 0.3,
                                                                "smoothness_spar": 0.3,
                                                                "smoothness_sigma": 1}},

            'n_neurons_of_hidden_layer_x_autoencoder': [1000, 500, 250, 125],  # 1000, 500, 250, 125
            'n_neurons_of_hidden_layer_x_discriminator': [500, 250, 125],  # 500, 250, 125
            'n_neurons_of_hidden_layer_x_discriminator_c': [1000, 1000],  # for semi-supervised
            'n_neurons_of_hidden_layer_x_discriminator_g': [1000, 1000],  # for semi-supervised

            'dropout_encoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_decoder': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator': [0.0, 0.0, 0.0, 0.0],
            'dropout_discriminator_c': [0.0, 0.0, 0.0],
            'dropout_discriminator_g': [0.0, 0.0, 0.0],

            'batch_normalization_encoder': ["post_activation", "post_activation", "post_activation", "post_activation",
                                            "post_activation"],
            'batch_normalization_decoder': ["post_activation", "post_activation", "post_activation", "post_activation",
                                            "post_activation"],
            'batch_normalization_discriminator': [None, None, None, None, None],
            # for semi-supervised:
            'batch_normalization_discriminator_c': ["post_activation", "post_activation", "post_activation"],
            'batch_normalization_discriminator_g': ["post_activation", "post_activation", "post_activation"],

            'bias_initializer_encoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                         "zeros_initializer", "zeros_initializer"],
            'bias_initializer_decoder': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                         "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator': ["zeros_initializer", "zeros_initializer", "zeros_initializer",
                                               "zeros_initializer"],
            'bias_initializer_discriminator_c': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],
            'bias_initializer_discriminator_g': ["zeros_initializer", "zeros_initializer", "zeros_initializer"],

            'bias_initializer_params_encoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_decoder': [{}, {}, {}, {}, {}],
            'bias_initializer_params_discriminator': [{}, {}, {}, {}],
            'bias_initializer_params_discriminator_c': [{}, {}, {}],
            'bias_initializer_params_discriminator_g': [{}, {}, {}],

            'weights_initializer_encoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_decoder': ["truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer", "truncated_normal_initializer",
                                            "truncated_normal_initializer"],
            'weights_initializer_discriminator': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                  "truncated_normal_initializer", "truncated_normal_initializer"],
            'weights_initializer_discriminator_c': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],
            'weights_initializer_discriminator_g': ["truncated_normal_initializer", "truncated_normal_initializer",
                                                    "truncated_normal_initializer"],

            'weights_initializer_params_encoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_decoder': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                   {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                         {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_c': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],
            'weights_initializer_params_discriminator_g': [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                                                           {"mean": 0, "stddev": 0.1}],

            'activation_function_encoder': ['relu', 'relu', 'relu', 'relu', 'linear'],
            'activation_function_decoder': ['sigmoid', 'relu', 'relu', 'relu', 'linear'],
            'activation_function_discriminator': ['relu', 'relu', 'relu', 'linear'],
            'activation_function_discriminator_c': ['relu', 'relu', 'linear'],  # for semi-supervised
            'activation_function_discriminator_g': ['relu', 'relu', 'linear'],  # for semi-supervised

            'decaying_learning_rate_name_autoencoder': "static",
            'decaying_learning_rate_name_discriminator': "static",
            'decaying_learning_rate_name_generator': "static",
            'decaying_learning_rate_name_discriminator_gaussian': "static",
            'decaying_learning_rate_name_discriminator_categorical': "static",
            'decaying_learning_rate_name_supervised_encoder': "static",

            'decaying_learning_rate_params_autoencoder': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_generator': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_gaussian': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_discriminator_categorical': {"learning_rate": 0.0001},
            'decaying_learning_rate_params_supervised_encoder': {"learning_rate": 0.0001},

            'optimizer_autoencoder': 'AdamOptimizer',
            'optimizer_discriminator': 'AdamOptimizer',
            'optimizer_generator': 'AdamOptimizer',
            'optimizer_discriminator_gaussian': 'AdamOptimizer',
            'optimizer_discriminator_categorical': 'AdamOptimizer',
            'optimizer_supervised_encoder': 'AdamOptimizer',
            'AdadeltaOptimizer_rho_autoencoder': 0.95, 'AdadeltaOptimizer_epsilon_autoencoder': 1e-08,
            'AdadeltaOptimizer_rho_discriminator': 0.95, 'AdadeltaOptimizer_epsilon_discriminator': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_gaussian': 0.95,
            'AdadeltaOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdadeltaOptimizer_rho_discriminator_categorical': 0.95, 'AdadeltaOptimizer_epsilon_categorical': 1e-08,
            'AdadeltaOptimizer_rho_supervised_encoder': 0.95, 'AdadeltaOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdadeltaOptimizer_rho_generator': 0.95, 'AdadeltaOptimizer_epsilon_generator': 1e-08,
            'AdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'AdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'AdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'AdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'MomentumOptimizer_momentum_autoencoder': 0.9, 'MomentumOptimizer_use_nesterov_autoencoder': False,
            'MomentumOptimizer_momentum_discriminator': 0.9, 'MomentumOptimizer_use_nesterov_discriminator': False,
            'MomentumOptimizer_momentum_discriminator_gaussian': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_gaussian': False,
            'MomentumOptimizer_momentum_discriminator_categorical': 0.9,
            'MomentumOptimizer_use_nesterov_discriminator_categorical': False,
            'MomentumOptimizer_momentum_supervised_encoder': 0.9,
            'MomentumOptimizer_use_nesterov_supervised_encoder': False,
            'MomentumOptimizer_momentum_generator': 0.9, 'MomentumOptimizer_use_nesterov_generator': False,
            'AdamOptimizer_beta1_autoencoder': 0.9, 'AdamOptimizer_beta2_autoencoder': 0.999,
            'AdamOptimizer_epsilon_autoencoder': 1e-08, 'AdamOptimizer_beta1_discriminator': 0.9,
            'AdamOptimizer_beta2_discriminator': 0.999, 'AdamOptimizer_epsilon_discriminator': 1e-08,
            'AdamOptimizer_beta1_discriminator_gaussian': 0.9,
            'AdamOptimizer_beta2_discriminator_gaussian': 0.999,
            'AdamOptimizer_epsilon_discriminator_gaussian': 1e-08,
            'AdamOptimizer_beta1_discriminator_categorical': 0.9,
            'AdamOptimizer_beta2_discriminator_categorical': 0.999,
            'AdamOptimizer_epsilon_discriminator_categorical': 1e-08,
            'AdamOptimizer_beta1_supervised_encoder': 0.9,
            'AdamOptimizer_beta2_supervised_encoder': 0.999, 'AdamOptimizer_epsilon_supervised_encoder': 1e-08,
            'AdamOptimizer_beta1_generator': 0.9, 'AdamOptimizer_beta2_generator': 0.999,
            'AdamOptimizer_epsilon_generator': 1e-08, 'FtrlOptimizer_learning_rate_power_autoencoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_gaussian': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_gaussian': 0.0,
            'FtrlOptimizer_learning_rate_power_discriminator_categorical': -0.5,
            'FtrlOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'FtrlOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_categorical': 0.0,
            'FtrlOptimizer_learning_rate_power_supervised_encoder': -0.5,
            'FtrlOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'FtrlOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_supervised_encoder': 0.0,
            'FtrlOptimizer_learning_rate_power_generator': -0.5,
            'FtrlOptimizer_initial_accumulator_value_generator': 0.1,
            'FtrlOptimizer_l1_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_regularization_strength_generator': 0.0,
            'FtrlOptimizer_l2_shrinkage_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalGradientDescentOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalGradientDescentOptimizer_l2_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_autoencoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_supervised_encoder': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_supervised_encoder': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_gaussian': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_gaussian': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator_categorical': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator_categorical': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_discriminator': 0.0,
            'ProximalAdagradOptimizer_initial_accumulator_value_generator': 0.1,
            'ProximalAdagradOptimizer_l1_regularization_strength_generator': 0.0,
            'ProximalAdagradOptimizer_l2_regularization_strength_generator': 0.0,
            'RMSPropOptimizer_decay_autoencoder': 0.9,
            'RMSPropOptimizer_momentum_autoencoder': 0.0,
            'RMSPropOptimizer_epsilon_autoencoder': 1e-10, 'RMSPropOptimizer_centered_autoencoder': False,
            'RMSPropOptimizer_decay_discriminator': 0.9, 'RMSPropOptimizer_momentum_discriminator': 0.0,
            'RMSPropOptimizer_epsilon_discriminator': 1e-10,
            'RMSPropOptimizer_centered_discriminator': False,
            'RMSPropOptimizer_decay_discriminator_gaussian': 0.9,
            'RMSPropOptimizer_momentum_discriminator_gaussian': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_gaussian': 1e-10,
            'RMSPropOptimizer_centered_discriminator_gaussian': False,
            'RMSPropOptimizer_decay_discriminator_categorical': 0.9,
            'RMSPropOptimizer_momentum_discriminator_categorical': 0.0,
            'RMSPropOptimizer_epsilon_discriminator_categorical': 1e-10,
            'RMSPropOptimizer_centered_discriminator_categorical': False,
            'RMSPropOptimizer_decay_supervised_encoder': 0.9, 'RMSPropOptimizer_momentum_supervised_encoder': 0.0,
            'RMSPropOptimizer_epsilon_supervised_encoder': 1e-10,
            'RMSPropOptimizer_centered_supervised_encoder': False,
            'RMSPropOptimizer_decay_generator': 0.9, 'RMSPropOptimizer_momentum_generator': 0.0,
            'RMSPropOptimizer_epsilon_generator': 1e-10, 'RMSPropOptimizer_centered_generator': False,
            'loss_function_discriminator': 'sigmoid_cross_entropy',
            'loss_function_discriminator_gaussian': 'sigmoid_cross_entropy',
            'loss_function_discriminator_categorical': 'sigmoid_cross_entropy',
            'loss_function_generator': 'sigmoid_cross_entropy'}


def get_default_parameters(selected_autoencoder, selected_dataset):
    """
    returns the default parameters based on the selected autoencoder and selected dataset
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param selected_dataset: dictionary holding the parameters needed to create the Autoencoder
    :return:
    """

    if selected_dataset == "MNIST":
        param_dict = get_default_parameters_mnist()
    elif selected_dataset == "SVHN":
        param_dict = get_default_parameters_svhn()
    elif selected_dataset == "cifar10":
        param_dict = get_default_parameters_cifar10()
    elif selected_dataset == "mass_spec":
        param_dict = get_default_parameters_mass_spec()
    elif selected_dataset == "custom":
        # TODO: implement
        raise NotImplementedError
    else:
        raise ValueError("Dataset " + selected_dataset + " not found.")

    param_dict["results_path"] = get_result_path_for_selected_autoencoder(selected_autoencoder)
    return param_dict


def create_network_topology(n_layers, init_n_neurons, n_neurons_decay_factor, n_decaying_layers):
    """
    creates a network topology based on the provided parameters
    :param n_layers: number of layers the network should have
    :param init_n_neurons: number of neurons the first layer should have
    :param n_neurons_decay_factor: by what factor the number of neurons in the suceeding layers should be reduced
    e.g. with a factor of 2: [3000, 3000, 3000] -> [3000, 1500, 750]
    :param n_decaying_layers: number of layers where the number of neurons should be reduced by the
    n_neurons_decay_factor
    :return:
    """
    random_network_topology = [init_n_neurons] * n_layers
    for decaying_layer in range(n_decaying_layers, 0, -1):
        random_network_topology[n_layers - decaying_layer] = \
            int(random_network_topology[n_layers - decaying_layer - 1] / n_neurons_decay_factor)

    return random_network_topology


def create_random_network_architectures(max_layers, init_n_neurons, n_neurons_decay_factors):
    """
    creates all combinations of max_layers and n_neurons_decay_factors; random network architectures with a certain
    number  of layers; n neurons and a decay factor for the number of neurons;
        e.g. max_layers=3, init_n_neurons=3000, n_neurons_decay_factors=[0.5, 2] results in:
            -   [[3000]
                [3000, 3000]
                [3000, 6000]
                [3000, 1500]
                [3000, 3000, 3000]
                [3000, 3000, 6000]
                [3000, 3000, 1500]
                [3000, 6000, 12000]
                [3000, 1500, 750]]
    :param max_layers: maximum amount of layers
    :param init_n_neurons: initial number of neurons
    :param n_neurons_decay_factors: list of floats; factor for the number of neurons in the preceding layer; e.g.:
        [0.5, 1.5, 2, 3]
    :return: list of lists with the respective network architecture; see above for an example
    """
    random_network_topologies = []

    for n_layers in range(1, max_layers + 1):
        # maximum number of layers with a reduced number of neurons compared to the preceding layers
        for n_decaying_layers in range(n_layers):
            # we only have to iterate over the n_neurons_decay_factors if we have at least one decaying layer
            if n_decaying_layers > 0:
                for n_neurons_decay_factor in n_neurons_decay_factors:
                    random_network_topologies.append(
                        create_network_topology(n_layers, init_n_neurons, n_neurons_decay_factor,
                                                n_decaying_layers))
            # otherwise we don't have any decaying layers
            else:
                random_network_topologies.append(
                    create_network_topology(n_layers, init_n_neurons, 1, n_decaying_layers))

    return random_network_topologies


def get_gridsearch_parameters(selected_autoencoder, selected_dataset, *args, **kwargs):
    """
    Performs a grid search for the provided parameters. If there are no parameters provided, it uses the hard coded
    parameters for grid search.
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param args:
    :param kwargs:
    :return:
    """

    # store verbose and the results_path in a local variable, so the value error below doesn't get raised..
    verbose = None
    if "verbose" in kwargs:
        if isinstance(kwargs["verbose"], list):
            verbose = kwargs["verbose"][0]
        else:
            verbose = kwargs["verbose"]
        del kwargs["verbose"]
    results_path = None
    if "results_path" in kwargs:
        results_path = kwargs["results_path"]
        del kwargs["results_path"]

    for element in kwargs.values():
        if not isinstance(element, list):
            print(element)

    # deal with illegal input
    if not all(isinstance(element, list) for element in kwargs.values()):
        raise ValueError("Key worded arguments must be provided as a list: "
                         "e.g. get_gridsearch_parameters(learning_rate=[0.5]) for single values or "
                         "get_gridsearch_parameters(learning_rate=[0.01, 0.01]) for multiple values.")

    """
    finding all parameter combinations and storing it in a list of dictionaries:
    """

    # get the default parameters
    param_dict = get_default_parameters(selected_autoencoder, selected_dataset)

    # set the verbose parameter properly
    if verbose:
        param_dict["verbose"] = verbose
    if results_path:
        param_dict["results_path"] = results_path

    # iterate over the variable names provided as parameters and set their value in the parameter dictionary as
    # defined above
    if args:
        for var_name in args:
            param_dict[var_name] = locals()[var_name]

    # iterate over the variable names provided as parameters and set their value in the parameter dictionary as
    # defined above
    if kwargs:
        for var_name in kwargs:
            param_dict[var_name] = kwargs[var_name]

    # holds the parameters which are by default selected for gridsearch
    default_params_selected_for_gridsearch = []

    # we don't have variable names provided, so we take all hard coded parameters for the grid search
    if not args and not kwargs:
        # those vars are always lists, so we need to ignore them
        local_vars_to_ignore = ["loss_functions", "param_dict_mnist", "optimizers", "autoencoder_optimizers",
                                "local_vars_to_ignore", "args", "kwargs", "default_params_selected_for_gridsearch",
                                "n_neurons_of_hidden_layer_x_autoencoder",
                                "n_neurons_of_hidden_layer_x_discriminator"]
        # check for hard coded grid search parameters (they are lists)
        for var_name in list(
                locals()):  # convert to list to avoid RuntimeError: dictionary changed during iteration
            # ignore the variables which are always lists
            if var_name not in local_vars_to_ignore:
                if type(locals()[var_name]) == list:
                    default_params_selected_for_gridsearch.append(var_name)
                else:
                    param_dict[var_name] = locals()[var_name]

    # get the parameters selected for gridsearch and store them in one list
    params_selected_for_gridsearch = list(args) + list(kwargs.keys())

    # these parameters are by default lists; so we can't use them for itertools.product
    params_default_as_list = ["n_neurons_of_hidden_layer_x_autoencoder",
                              "n_neurons_of_hidden_layer_x_discriminator"]

    # get all the parameter values and store them in a list of lists e.g. [[0.1, 0.2, 0.3], [1.0, 5.0, 9.0]]
    param_values = [param_dict[param_selected_for_gridsearch] for param_selected_for_gridsearch
                    in params_selected_for_gridsearch]

    # add the  parameters selected for gridsearch by default
    params_selected_for_gridsearch += default_params_selected_for_gridsearch

    # add their values to the param_values list
    for default_param_selected_for_gridsearch in default_params_selected_for_gridsearch:
        param_values.append(locals()[default_param_selected_for_gridsearch])

    # stores all the resulting parameter combinations
    all_final_parameter_combinations_list = []

    # get all combinations
    parameter_value_combinations = list(itertools.product(*param_values))

    # iterate over the combinations ..
    for parameter_value_combination in parameter_value_combinations:
        for i, param_value in enumerate(parameter_value_combination):
            # .. set the param_dict_mnist accordingly ..
            param_dict[params_selected_for_gridsearch[i]] = param_value
        # .. and add them to the list
        all_final_parameter_combinations_list.append(copy.deepcopy(param_dict))

    return all_final_parameter_combinations_list


def randomize_params_for_tf_initializer(initializer_name):
    if initializer_name == "constant_initializer":
        # value: A Python scalar, list or tuple of values, or a N-dimensional numpy array. All elements of the
        # initialized variable will be set to the corresponding value in the value argument.
        return {"value": draw_from_np_distribution(distribution_name="uniform", low=-0.3, high=0.3, return_type="float")}
    elif initializer_name == "random_normal_initializer":
        # mean: a python scalar or a scalar tensor. Mean of the random values to generate.
        # stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
        return {"mean": draw_from_np_distribution(distribution_name="uniform", low=-0.3, high=0.3, return_type="float"),
                "stddev": draw_from_np_distribution(distribution_name="uniform", low=-0.5, high=0.5, return_type="float")}
    elif initializer_name == "truncated_normal_initializer":
        # mean: a python scalar or a scalar tensor. Mean of the random values to generate.
        # stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
        return {"mean": draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float"),
                "stddev": draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.5, return_type="float")}
    elif initializer_name == "random_uniform_initializer":
        # minval: A python scalar or a scalar tensor. Lower bound of the range of random values to generate.
        # maxval: A python scalar or a scalar tensor. Upper bound of the range of random values to generate.
        #         Defaults to 1 for float types.
        return {"minval": draw_from_np_distribution(distribution_name="uniform", low=-0.3, high=0.0, return_type="float"),
                "maxval": draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float")}
    elif initializer_name == "uniform_unit_scaling_initializer":
        # factor: Float. A multiplicative factor by which the values will be scaled.
        return {"factor": draw_from_np_distribution(distribution_name="uniform", low=-0.5, high=0.5, return_type="float")}
    elif initializer_name == "zeros_initializer":
        return {}
    elif initializer_name == "ones_initializer":
        return {}
    elif initializer_name == "orthogonal_initializer":
        # gain: multiplicative factor to apply to the orthogonal matrix
        return {"gain": draw_from_np_distribution(distribution_name="uniform", low=-0.5, high=0.5, return_type="float")}
    else:
        raise ValueError("Invalid initializer_name! " + initializer_name + " is invalid")


def randomize_params_for_decaying_learning_rate(decaying_learning_rate_name):

    if decaying_learning_rate_name == "exponential_decay":
        """
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        """
        return {"decay_steps": draw_from_np_distribution(distribution_name="uniform", low=10000, high=1000000, return_type="float"),
                "decay_rate": draw_from_np_distribution(distribution_name="uniform", low=0.94, high=0.98, return_type="float"),
                "staircase": random.choice([True, False])}

    elif decaying_learning_rate_name == "inverse_time_decay":
        """
        staircase=False:
            decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
        staircase=True:
            decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
        """
        return {"decay_steps": draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.2, return_type="float"),
                "decay_rate": draw_from_np_distribution(distribution_name="uniform", low=0.3, high=0.7, return_type="float"),
                "staircase": random.choice([True, False])}

    elif decaying_learning_rate_name == "natural_exp_decay":
        """
        decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        """
        return {"decay_steps": draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.2, return_type="float"),
                "decay_rate": draw_from_np_distribution(distribution_name="uniform", low=0.3, high=0.7, return_type="float"),
                "staircase": random.choice([True, False])}

    elif decaying_learning_rate_name == "piecewise_constant":
        """
        Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5 for steps 100001 to 110000, and 0.1 for
        any additional steps:
            boundaries = [100000, 110000]
            values = [1.0, 0.5, 0.1]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        """
        n_boundary_steps = draw_from_np_distribution(distribution_name="uniform", low=1, high=5, return_type="int")
        distance_boundary_steps = draw_from_np_distribution(distribution_name="uniform", low=100, high=2000, return_type="float")
        learning_rate_decay_factor = draw_from_np_distribution(distribution_name="uniform", low=0.1, high=0.5, return_type="float")
        learning_rate_values = [learning_rate_decay_factor] * (n_boundary_steps + 1)

        return {"boundaries": [distance_boundary_steps * i for i in range(1, n_boundary_steps)],
                "values": [np.product(learning_rate_values[:i]) for i in range(1, n_boundary_steps + 1)]}

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
        return {"decay_steps": draw_from_np_distribution(distribution_name="uniform", low=10000, high=1000000, return_type="float"),
                "end_learning_rate": draw_from_np_distribution(distribution_name="uniform", low=0.000001, high=0.0001, return_type="float"),
                "power": draw_from_np_distribution(distribution_name="uniform", low=0.95, high=1.05, return_type="float"),
                "cycle": random.choice([True, False])}

    elif decaying_learning_rate_name == "static":
        """
        Static learning rate.
        """
        return {}

    else:
        raise ValueError(decaying_learning_rate_name, "is not a valid value for this variable.")


def get_randomized_parameters(*args, selected_autoencoder, selected_dataset, **kwargs):
    """
    returns randomized values for the specified parameters; otherwise the default values
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param args: string or list of strings with the parameters which should be randomized; if empty randomizes
    all parameters
    :return: dictionary: {'parameter1': parameter1_value, 'parameter2': parameter2_value}
    """

    # train duration
    batch_size = draw_from_np_distribution(distribution_name="uniform", low=50, high=500, return_type="int")
    n_epochs = 10  # TODO: probably doesn't make really sense ..
    z_dim = draw_from_np_distribution(distribution_name="uniform", low=2, high=100, return_type="int")
    n_labeled = draw_from_np_distribution(distribution_name="uniform", low=1000, high=10000, return_type="int")  # for semi-supervised

    # network architecture
    n_neurons_of_hidden_layer_x_autoencoder = random.choice(
        create_random_network_architectures(max_layers=3, init_n_neurons=2000, n_neurons_decay_factors=[0.5, 2]))
    n_neurons_of_hidden_layer_x_discriminator = random.choice(
        create_random_network_architectures(max_layers=3, init_n_neurons=2000, n_neurons_decay_factors=[0.5, 2]))
    n_neurons_of_hidden_layer_x_discriminator_c = random.choice(
        create_random_network_architectures(max_layers=3, init_n_neurons=2000, n_neurons_decay_factors=[0.5, 2]))
    n_neurons_of_hidden_layer_x_discriminator_g = random.choice(
        create_random_network_architectures(max_layers=3, init_n_neurons=2000, n_neurons_decay_factors=[0.5, 2]))

    # check whether the network architecture is selected for random search..
    for subnetwork_name in ["n_neurons_of_hidden_layer_x_autoencoder", "n_neurons_of_hidden_layer_x_discriminator",
                            "n_neurons_of_hidden_layer_x_discriminator_c",
                            "n_neurons_of_hidden_layer_x_discriminator_g"]:
        # if it's not selected, we need to get the default parameters for the current data set; so the variables which
        # are depending on the network architecture like the bias initialization do not fail
        if not(subnetwork_name in args or subnetwork_name in kwargs):
            if subnetwork_name == "n_neurons_of_hidden_layer_x_autoencoder":
                n_neurons_of_hidden_layer_x_autoencoder = get_default_parameters(selected_autoencoder, selected_dataset)[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator":
                n_neurons_of_hidden_layer_x_discriminator = get_default_parameters(selected_autoencoder, selected_dataset)[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator_c":
                n_neurons_of_hidden_layer_x_discriminator_c = get_default_parameters(selected_autoencoder, selected_dataset)[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator_g":
                n_neurons_of_hidden_layer_x_discriminator_g = get_default_parameters(selected_autoencoder, selected_dataset)[subnetwork_name]
        # we need to update the local variable, so that the variables which are depending on the network architecture
        # like the bias initialization do not fail
        elif subnetwork_name in kwargs:
            if subnetwork_name == "n_neurons_of_hidden_layer_x_autoencoder":
                n_neurons_of_hidden_layer_x_autoencoder = kwargs[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator":
                n_neurons_of_hidden_layer_x_discriminator = kwargs[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator_c":
                n_neurons_of_hidden_layer_x_discriminator_c = kwargs[subnetwork_name]
            elif subnetwork_name == "n_neurons_of_hidden_layer_x_discriminator_g":
                n_neurons_of_hidden_layer_x_discriminator_g = kwargs[subnetwork_name]

    n_layers_autoencoder = len(n_neurons_of_hidden_layer_x_autoencoder) + 1
    n_layers_discriminator = len(n_neurons_of_hidden_layer_x_discriminator) + 1
    n_layers_discriminator_c = len(n_neurons_of_hidden_layer_x_discriminator_c) + 1
    n_layers_discriminator_g = len(n_neurons_of_hidden_layer_x_discriminator_g) + 1

    # randomize dropout
    dropout_encoder = [draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float") for i in range(n_layers_autoencoder)]
    dropout_decoder = [draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float") for i in range(n_layers_autoencoder)]
    dropout_discriminator = [draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float") for i in range(n_layers_discriminator)]
    dropout_discriminator_c = [draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float") for i in range(n_layers_discriminator_c)]
    dropout_discriminator_g = [draw_from_np_distribution(distribution_name="uniform", low=0.0, high=0.3, return_type="float") for i in range(n_layers_discriminator_g)]

    # randomize batch normalization
    batch_normalization_options = ["pre_activation", "post_activation", None]
    batch_normalization_encoder = [random.choice(batch_normalization_options) for i in range(n_layers_autoencoder)]
    batch_normalization_decoder = [random.choice(batch_normalization_options) for i in range(n_layers_autoencoder)]
    batch_normalization_discriminator = [random.choice(batch_normalization_options) for i in range(n_layers_discriminator)]
    batch_normalization_discriminator_c = [random.choice(batch_normalization_options) for i in range(n_layers_discriminator_c)]
    batch_normalization_discriminator_g = [random.choice(batch_normalization_options) for i in range(n_layers_discriminator_g)]

    # randomize bias initialization
    bias_initializer_options = ["constant_initializer", "random_normal_initializer", "truncated_normal_initializer",
                           "random_uniform_initializer", "uniform_unit_scaling_initializer", "zeros_initializer",
                           "ones_initializer"]
    bias_initializer_encoder = [random.choice(bias_initializer_options) for i in range(n_layers_autoencoder)]
    # bias_initializer_encoder = ["orthogonal_initializer"] * n_layers_autoencoder
    bias_initializer_decoder = [random.choice(bias_initializer_options) for i in range(n_layers_autoencoder)]
    bias_initializer_discriminator = [random.choice(bias_initializer_options) for i in range(n_layers_discriminator)]
    bias_initializer_discriminator_c = [random.choice(bias_initializer_options) for i in range(n_layers_discriminator_c)]
    bias_initializer_discriminator_g = [random.choice(bias_initializer_options) for i in range(n_layers_discriminator_g)]

    bias_initializer_params_encoder = [randomize_params_for_tf_initializer(initializer) for initializer in bias_initializer_encoder]
    bias_initializer_params_decoder = [randomize_params_for_tf_initializer(initializer) for initializer in bias_initializer_decoder]
    bias_initializer_params_discriminator = [randomize_params_for_tf_initializer(initializer) for initializer in bias_initializer_discriminator]
    bias_initializer_params_discriminator_c = [randomize_params_for_tf_initializer(initializer) for initializer in bias_initializer_discriminator_c]
    bias_initializer_params_discriminator_g = [randomize_params_for_tf_initializer(initializer) for initializer in bias_initializer_discriminator_g]

    # randomize weight initialization
    weights_initializer_options = ["constant_initializer", "random_normal_initializer", "truncated_normal_initializer",
                                   "random_uniform_initializer", "uniform_unit_scaling_initializer",
                                   "zeros_initializer", "ones_initializer"]
    weights_initializer_encoder = [random.choice(weights_initializer_options) for i in range(n_layers_autoencoder)]
    weights_initializer_decoder = [random.choice(weights_initializer_options) for i in range(n_layers_autoencoder)]
    weights_initializer_discriminator = [random.choice(weights_initializer_options) for i in range(n_layers_autoencoder)]
    weights_initializer_discriminator_c = [random.choice(weights_initializer_options) for i in range(n_layers_autoencoder)]
    weights_initializer_discriminator_g = [random.choice(weights_initializer_options) for i in range(n_layers_autoencoder)]

    weights_initializer_params_encoder = [randomize_params_for_tf_initializer(initializer) for initializer in weights_initializer_encoder]
    weights_initializer_params_decoder = [randomize_params_for_tf_initializer(initializer) for initializer in weights_initializer_decoder]
    weights_initializer_params_discriminator = [randomize_params_for_tf_initializer(initializer) for initializer in weights_initializer_discriminator]
    weights_initializer_params_discriminator_c = [randomize_params_for_tf_initializer(initializer) for initializer in weights_initializer_discriminator_c]
    weights_initializer_params_discriminator_g = [randomize_params_for_tf_initializer(initializer) for initializer in weights_initializer_discriminator_g]

    # activation functions
    activation_function_options = ["relu", "relu6", "crelu", "elu", "softplus", "softsign", "sigmoid", "tanh",
                                   "leaky_relu", "linear"]

    activation_function_encoder = [random.choice(activation_function_options) for i in range(n_layers_autoencoder)]
    activation_function_decoder = [random.choice(activation_function_options) for i in range(n_layers_autoencoder)]
    activation_function_discriminator = [random.choice(activation_function_options) for i in range(n_layers_discriminator)]
    activation_function_discriminator_c = [random.choice(activation_function_options) for i in range(n_layers_discriminator_c)]
    activation_function_discriminator_g = [random.choice(activation_function_options) for i in range(n_layers_discriminator_g)]

    # decaying learning rates
    learning_rate_options = ["exponential_decay", "inverse_time_decay", "natural_exp_decay", "piecewise_constant",
                             "polynomial_decay", "static"]

    decaying_learning_rate_name_autoencoder = random.choice(learning_rate_options)
    decaying_learning_rate_name_discriminator = random.choice(learning_rate_options)
    decaying_learning_rate_name_generator = random.choice(learning_rate_options)
    decaying_learning_rate_name_discriminator_gaussian = random.choice(learning_rate_options)
    decaying_learning_rate_name_discriminator_categorical = random.choice(learning_rate_options)
    decaying_learning_rate_name_supervised_encoder = random.choice(learning_rate_options)

    decaying_learning_rate_params_autoencoder = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_autoencoder)
    decaying_learning_rate_params_discriminator = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_discriminator)
    decaying_learning_rate_params_generator = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_generator)
    decaying_learning_rate_params_discriminator_gaussian = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_discriminator_gaussian)
    decaying_learning_rate_params_discriminator_categorical = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_discriminator_categorical)
    decaying_learning_rate_params_supervised_encoder = randomize_params_for_decaying_learning_rate(decaying_learning_rate_name_supervised_encoder)

    # available optimizers:
    autoencoder_optimizers = ["AdamOptimizer",
                              "RMSPropOptimizer"]

    optimizers = ["GradientDescentOptimizer",  # autoencoder part not working
                  "AdadeltaOptimizer",  # autoencoder part not working
                  "AdagradOptimizer",  # autoencoder part not working
                  "MomentumOptimizer",  # autoencoder part not working
                  "AdamOptimizer",
                  # "FtrlOptimizer",  # autoencoder part not working; optimizer slow + bad results
                  "ProximalGradientDescentOptimizer",  # autoencoder part not working
                  "ProximalAdagradOptimizer",  # autoencoder part not working
                  "RMSPropOptimizer"]

    optimizer_autoencoder = random.choice(autoencoder_optimizers)
    optimizer_discriminator = random.choice(optimizers)
    optimizer_discriminator_gaussian = random.choice(optimizers)
    optimizer_discriminator_categorical = random.choice(optimizers)
    optimizer_supervised_encoder = random.choice(optimizers)
    optimizer_generator = random.choice(optimizers)

    """
    https://www.tensorflow.org/api_guides/python/train#Optimizers
    parameters for optimizers:
    """

    # GradientDescentOptimizer:
    #   - learning rate:

    # AdadeltaOptimizer():
    #   - learning rate: default: 0.01
    #   - rho: decay rate; default: 0.95
    #   - epsilon: A constant epsilon used to better conditioning the grad update; default: 1e-08
    AdadeltaOptimizer_rho_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    AdadeltaOptimizer_rho_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    AdadeltaOptimizer_rho_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    AdadeltaOptimizer_rho_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    AdadeltaOptimizer_rho_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    AdadeltaOptimizer_rho_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.9, high=1.0, return_type="float")
    AdadeltaOptimizer_epsilon_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-09, high=1e-07, return_type="float")

    # AdagradOptimizer
    #   - learning rate
    #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
    #   default: 0.1
    AdagradOptimizer_initial_accumulator_value_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")
    AdagradOptimizer_initial_accumulator_value_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")
    AdagradOptimizer_initial_accumulator_value_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")
    AdagradOptimizer_initial_accumulator_value_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")
    AdagradOptimizer_initial_accumulator_value_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")
    AdagradOptimizer_initial_accumulator_value_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.2, return_type="float")

    # MomentumOptimizer
    #   - learning rate
    #   - momentum: A Tensor or a floating point value. The momentum.
    #   - use_nesterov: If True use Nesterov Momentum; default: False http://proceedings.mlr.press/v28/sutskever13.pdf
    MomentumOptimizer_momentum_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_autoencoder = random.choice([True, False])

    MomentumOptimizer_momentum_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_discriminator = random.choice([True, False])

    MomentumOptimizer_momentum_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_discriminator_gaussian = random.choice([True, False])

    MomentumOptimizer_momentum_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_discriminator_categorical = random.choice([True, False])

    MomentumOptimizer_momentum_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_supervised_encoder = random.choice([True, False])

    MomentumOptimizer_momentum_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    MomentumOptimizer_use_nesterov_generator = random.choice([True, False])

    # AdamOptimizer
    #   - learning rate; default: 0.001; The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3)
    #       results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning
    #       right down during training
    #   - beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
    #   default: 0.9
    #   - beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
    #   default: 0.999
    #   - epsilon: A small constant for numerical stability. default: 1e-08
    AdamOptimizer_beta1_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    AdamOptimizer_beta1_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    AdamOptimizer_beta1_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    AdamOptimizer_beta1_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    AdamOptimizer_beta1_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    AdamOptimizer_beta1_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    AdamOptimizer_beta2_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.99, high=1.0, return_type="float")
    AdamOptimizer_epsilon_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-07, high=1e-09, return_type="float")

    # FtrlOptimizer
    #   - learning rate
    #   - learning rate power: A float value, must be less or equal to zero. default: -0.5
    #   - initial_accumulator_value: The starting value for accumulators. Only positive values are allowed. default: 0.1
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_shrinkage_regularization_strength: A float value, must be greater than or equal to zero. This differs from
    #   L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
    #   default: 0.0
    FtrlOptimizer_learning_rate_power_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    FtrlOptimizer_learning_rate_power_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    FtrlOptimizer_learning_rate_power_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    FtrlOptimizer_learning_rate_power_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    FtrlOptimizer_learning_rate_power_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    FtrlOptimizer_learning_rate_power_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=-1, high=0, return_type="float")
    FtrlOptimizer_initial_accumulator_value_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    FtrlOptimizer_l1_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    FtrlOptimizer_l2_shrinkage_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    # ProximalGradientDescentOptimizer
    #   - learning rate
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalGradientDescentOptimizer_l1_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalGradientDescentOptimizer_l1_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalGradientDescentOptimizer_l2_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    # ProximalAdagradOptimizer
    #   - learning rate
    #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
    #   default: 0.1
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    ProximalAdagradOptimizer_initial_accumulator_value_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalAdagradOptimizer_initial_accumulator_value_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalAdagradOptimizer_initial_accumulator_value_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalAdagradOptimizer_initial_accumulator_value_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalAdagradOptimizer_initial_accumulator_value_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    ProximalAdagradOptimizer_initial_accumulator_value_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.01, high=0.3, return_type="float")
    ProximalAdagradOptimizer_l1_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")
    ProximalAdagradOptimizer_l2_regularization_strength_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.1, return_type="float")

    # RMSPropOptimizer
    #   - learning rate
    #   - decay: Discounting factor for the history/coming gradient; default: 0.9
    #   - momentum: A scalar tensor; default: 0.0.
    #   - epsilon:  Small value to avoid zero denominator.; default: 1e-10
    #   - centered: If True, gradients are normalized by the estimated variance of the gradient; if False, by the
    #   uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms
    #   of computation and memory. Defaults to False.
    RMSPropOptimizer_decay_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_autoencoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_autoencoder = random.choice([True, False])

    RMSPropOptimizer_decay_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_discriminator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_discriminator = random.choice([True, False])

    RMSPropOptimizer_decay_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_discriminator_gaussian = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_discriminator_gaussian = random.choice([True, False])

    RMSPropOptimizer_decay_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_discriminator_categorical = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_discriminator_categorical = random.choice([True, False])

    RMSPropOptimizer_decay_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_supervised_encoder = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_supervised_encoder = random.choice([True, False])

    RMSPropOptimizer_decay_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0.8, high=1.0, return_type="float")
    RMSPropOptimizer_momentum_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=0, high=0.2, return_type="float")
    RMSPropOptimizer_epsilon_generator = \
        draw_from_np_distribution(distribution_name="uniform", low=1e-9, high=1e-11, return_type="float")
    RMSPropOptimizer_centered_generator = random.choice([True, False])

    # available loss functions
    loss_functions = ["hinge_loss",
                      "mean_squared_error",
                      "sigmoid_cross_entropy",
                      "softmax_cross_entropy"]

    # loss function for discriminator
    loss_function_discriminator = random.choice(loss_functions)
    loss_function_discriminator_gaussian = random.choice(loss_functions)
    loss_function_discriminator_categorical = random.choice(loss_functions)
    # loss function for generator
    loss_function_generator = random.choice(loss_functions)

    # get the default parameters
    param_dict = get_default_parameters(selected_autoencoder, selected_dataset)

    # iterate over the variable names provided as parameters and set their value as random defined above
    if args:
        for var_name in args:
            param_dict[var_name] = locals()[var_name]

    # iterate over the variable names provided as keyword parameters and set their value accordingly
    if kwargs:
        for var_name in kwargs:
            # we have a dictionary, so we want to draw from the respective distribution
            if isinstance(kwargs[var_name], dict):
                # if the dictionary has a key "distribution_name", we want to draw from a numpy distribution..
                if kwargs[var_name].get("distribution_name"):
                    param_dict[var_name] = draw_from_np_distribution(**kwargs[var_name])
                # .. if not, the parameter is simply stored as a dictionary
                else:
                    param_dict[var_name] = kwargs[var_name]
            else:
                param_dict[var_name] = kwargs[var_name]

    if not kwargs and not args:
        local_vars_to_ignore = ["loss_functions", "param_dict", "optimizers", "autoencoder_optimizers",
                                "local_vars_to_ignore", "learning_rate_options", "activation_function_options",
                                "weights_initializer_options", "bias_initializer_options", "n_layers_autoencoder",
                                "n_layers_discriminator", "n_layers_discriminator_c", "n_layers_discriminator_g",
                                "args", "kwargs"]
        for var_name in list(
                locals()):  # convert to list to avoid RuntimeError: dictionary changed during iteration
            if var_name not in local_vars_to_ignore:
                param_dict[var_name] = locals()[var_name]

    return param_dict
