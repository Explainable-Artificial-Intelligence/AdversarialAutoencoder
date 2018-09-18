import json
import os
import numpy as np

from autoencoders.AAE_LearningPriorSupervised import LearningPriorsAdversarialAutoencoderSupervised
from autoencoders.AAE_LearningPriorUnsupervised import LearningPriorsAdversarialAutoencoderUnsupervised
from autoencoders.AAE_LearningPriorSameTopology import LearningPriorsAdversarialAutoencoderSameTopology
from autoencoders.DimensionalityReductionAdversarialAutoencoder import DimensionalityReductionAdversarialAutoencoder
from autoencoders.IncorporatingLabelInformationAdversarialAutoencoder import \
    IncorporatingLabelInformationAdversarialAutoencoder
from autoencoders.SemiSupervisedAdversarialAutoencoder import SemiSupervisedAdversarialAutoencoder
from autoencoders.SupervisedAdversarialAutoencoder import SupervisedAdversarialAutoencoder
from autoencoders.UnsupervisedAdversarialAutoencoder import UnsupervisedAdversarialAutoencoder
from autoencoders.UnsupervisedClusteringAdversarialAutoencoder import UnsupervisedClusteringAdversarialAutoencoder
from util.AdversarialAutoencoderParameters import get_default_parameters_mnist, \
    get_result_path_for_selected_autoencoder, get_default_parameters_svhn, get_default_parameters_mass_spec, \
    create_random_network_architectures, get_default_parameters, randomize_params_for_tf_initializer
from util.TuningFunctions import do_randomsearch, do_gridsearch, init_aae_with_params_file, get_params_from_params_file


def param_search_incorporating_label_information():

    params = get_default_parameters_mnist()
    params["selected_dataset"] = "MNIST"

    params["z_dim"] = 2
    params["verbose"] = True
    params["selected_autoencoder"] = "IncorporatingLabelInformation"
    params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
    params["summary_image_frequency"] = 10
    params["n_epochs"] = 751
    params["batch_normalization_encoder"] = [None, None, None, None, None]
    params["batch_normalization_decoder"] = [None, None, None, None, None]
    params["batch_normalization_discriminator"] = [None, None, None, None, None]

    params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
    params["n_neurons_of_hidden_layer_x_discriminator_c"] = [1000, 1000]
    params["n_neurons_of_hidden_layer_x_discriminator_g"] = [1000, 1000]

    params["activation_function_encoder"] = ['relu', 'relu', 'linear']
    params["activation_function_decoder"] = ['relu', 'relu', 'sigmoid']
    params["activation_function_discriminator_c"] = ['relu', 'relu', 'linear']
    params["activation_function_discriminator_g"] = ['relu', 'relu', 'linear']

    params["dropout_encoder"] = [0.0, 0.0, 0.0]
    params["dropout_decoder"] = [0.0, 0.0, 0.0]
    params["dropout_discriminator_c"] = [0.0, 0.0, 0.0]
    params["dropout_discriminator_g"] = [0.0, 0.0, 0.0]

    """
    try static learning rate:
    """
    params['decaying_learning_rate_name_autoencoder'] = "static"
    params['decaying_learning_rate_name_discriminator'] = "static"
    params['decaying_learning_rate_name_generator'] = "static"

    params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.001}
    params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.001}
    params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.001}

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()

    params['decaying_learning_rate_name_autoencoder'] = "static"
    params['decaying_learning_rate_name_discriminator'] = "static"
    params['decaying_learning_rate_name_generator'] = "static"

    params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
    params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
    params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()

    """
    try several piecewise constant learning rates:
    """
    params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
    params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
    params['decaying_learning_rate_name_generator'] = "piecewise_constant"

    params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}
    params["decaying_learning_rate_params_discriminator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}
    params["decaying_learning_rate_params_generator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()

    # next combination:
    params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
    params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
    params['decaying_learning_rate_name_generator'] = "piecewise_constant"

    params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [250], "values": [0.0001, 0.00001]}
    params["decaying_learning_rate_params_discriminator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}
    params["decaying_learning_rate_params_generator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()

    # next combination:
    params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
    params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
    params['decaying_learning_rate_name_generator'] = "piecewise_constant"

    params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [250, 450], "values": [0.0001, 0.00001, 0.000001]}
    params["decaying_learning_rate_params_discriminator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}
    params["decaying_learning_rate_params_generator"] = {"boundaries": [50, 250, 450], "values": [0.1, 0.01, 0.001, 0.0001]}

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()

    """
    try other params for the adam optimizer
    """
    params['decaying_learning_rate_name_autoencoder'] = "static"
    params['decaying_learning_rate_name_discriminator'] = "static"
    params['decaying_learning_rate_name_generator'] = "static"

    params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.001}
    params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.001}
    params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.001}

    params["AdamOptimizer_beta1_autoencoder"] = 0.5
    params["AdamOptimizer_beta1_discriminator"] = 0.5
    params["AdamOptimizer_beta1_generator"] = 0.5

    aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
    aae.train(True)
    aae.reset_graph()


def param_search_mass_spec_data():

    for i in [2, 8, 16, 50, 100, 150, 200, 250, 350, 500, 1000]:
        do_randomsearch(1, "bias_initializer_encoder", "bias_initializer_decoder", "bias_initializer_params_encoder",
                        "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                        "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                        selected_autoencoder="Unsupervised", z_dim=i,
                        selected_dataset="mass_spec", n_epochs=5001, summary_image_frequency=250,
                        AdamOptimizer_beta1_autoencoder=0.5)

    return

    do_randomsearch(10, "bias_initializer_encoder", "bias_initializer_decoder",  "bias_initializer_params_encoder",
                    "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                    "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                    selected_autoencoder="Unsupervised", z_dim=2,
                    selected_dataset="mass_spec", n_epochs=3, summary_image_frequency=500,
                    AdamOptimizer_beta1_autoencoder=0.5)

    return

    do_randomsearch(500, "bias_initializer_encoder", "bias_initializer_decoder",  "bias_initializer_params_encoder",
                    "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                    "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                    selected_autoencoder="Unsupervised", z_dim=10,
                    selected_dataset="mass_spec", n_epochs=5001,
                    AdamOptimizer_beta1_autoencoder=0.5)

    do_randomsearch(500, "bias_initializer_encoder", "bias_initializer_decoder",  "bias_initializer_params_encoder",
                    "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                    "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                    selected_autoencoder="Unsupervised", z_dim=50,
                    selected_dataset="mass_spec", n_epochs=5001,
                    AdamOptimizer_beta1_autoencoder=0.5)

    do_randomsearch(500, "bias_initializer_encoder", "bias_initializer_decoder",  "bias_initializer_params_encoder",
                    "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                    "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                    selected_autoencoder="Unsupervised", z_dim=100,
                    selected_dataset="mass_spec", n_epochs=5001,
                    AdamOptimizer_beta1_autoencoder=0.5)

    do_randomsearch(500, "bias_initializer_encoder", "bias_initializer_decoder",  "bias_initializer_params_encoder",
                    "bias_initializer_params_decoder", "weights_initializer_encoder", "weights_initializer_decoder",
                    "weights_initializer_params_encoder", "weights_initializer_params_decoder",
                    selected_autoencoder="Unsupervised", z_dim=150,
                    selected_dataset="mass_spec", n_epochs=5001,
                    AdamOptimizer_beta1_autoencoder=0.5)


def try_mass_spec_parameter_combinations():

    params = get_default_parameters_mass_spec()
    params["summary_image_frequency"] = 5
    params["n_epochs"] = 11

    # peak_encoding
    available_peak_encodings = ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "raw"]

    # datasubset:
    available_datasubsets = ["identified", "unidentified", None]

    # smoothing_method
    available_smoothing_methods = ["lowess", "gaussian_filter"]

    normalize_data = [True, False]
    charge = [None, "2", "3", "4"]          # "1" has too few datapoints
    include_molecular_weight_in_encoding = [True, False]
    include_charge_in_encoding = [True, False]

    for i in range(1):

        # params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": np.random.choice(available_peak_encodings),
        #                                        "use_smoothed_intensities": True, "data_subset": np.random.choice(available_datasubsets),
        #                                        "n_peaks_to_keep": 50, "max_intensity_value": 2000,
        #                                        "max_mz_value": 2000, "charge": np.random.choice(charge), "normalize_data": np.random.choice(normalize_data),
        #                                        "include_molecular_weight_in_encoding": np.random.choice(include_molecular_weight_in_encoding),
        #                                        "include_charge_in_encoding": np.random.choice(include_charge_in_encoding),
        #                                        "smoothness_params": {"smoothing_method": np.random.choice(available_smoothing_methods),
        #                                                              "smoothness_frac": 0.3,
        #                                                              "smoothness_sigma": 1,
        #                                                              "smoothing_n_gaussians": 15}}

        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "only_intensities",
                                               "use_smoothed_intensities": True, "data_subset": None,
                                               "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                               "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                               "include_molecular_weight_in_encoding": False,
                                               "include_charge_in_encoding": False,
                                               "smoothness_params": {"smoothing_method": "lowess",
                                                                     "smoothness_frac": 0.3,
                                                                     "smoothness_sigma": 1,
                                                                     "smoothing_n_gaussians": 15}}

        # TODO: does not work (labels cant be converted to one hot labels for some reason..)
        # params["mass_spec_data_properties"] = {'organism_name': 'yeast', 'peak_encoding': 'only_mz_charge_label', 'use_smoothed_intensities': True, 'data_subset': 'identified', 'n_peaks_to_keep': 50, 'max_intensity_value': 2000, 'max_mz_value': 2000, 'charge': '2', 'normalize_data': False, 'include_molecular_weight_in_encoding': False, 'include_charge_in_encoding': True, 'smoothness_params': {'smoothing_method': 'gaussian_filter', 'smoothness_frac': 0.3, 'smoothness_sigma': 1, 'smoothing_n_gaussians': 15}}


        print("parameter combination:")
        print(params["mass_spec_data_properties"])

        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
            params["input_dim_y"] = 1000
        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

        params["input_dim_x"] = 1
        params["n_classes"] = 2
        params["z_dim"] = 5

        params["selected_dataset"] = "mass_spec"

        params["only_train_autoencoder"] = True

        params["verbose"] = True
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.train(True)
        aae.reset_graph()


def param_search_weight_bias_initializers():

    for i in range(1):

        params = get_default_parameters_mass_spec()

        params["summary_image_frequency"] = 100
        params["n_epochs"] = 101

        params["mz_loss_factor"] = 1
        params["intensity_loss_factor"] = 1

        # peak_encoding
        ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
         "only_intensities_distance", "raw", "raw_intensities_sqrt", "raw_sqrt"]

        # datasubset:
        ["identified", "unidentified", None]

        # smoothing_method
        ["loess", "gaussian_filter", "spline"]

        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "raw_sqrt",
                                               "use_smoothed_intensities": False, "data_subset": None,
                                               "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                               "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                               "include_molecular_weight_in_encoding": False,
                                               "include_charge_in_encoding": False,
                                               "smoothness_params": {"smoothing_method": "spline",
                                                                     "smoothness_frac": 0.0,
                                                                     "smoothness_spar": 0.1,
                                                                     "smoothness_sigma": 1}}

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [250, 250, 250]
        n_layers_autoencoder = len(params["n_neurons_of_hidden_layer_x_autoencoder"]) + 1

        initializer_options = ["constant_initializer", "random_normal_initializer", "truncated_normal_initializer",
                                    "random_uniform_initializer", "uniform_unit_scaling_initializer",
                                    "zeros_initializer",
                                    "ones_initializer"]

        # pick bias initializer randomly
        bias_initializer_encoder = [np.random.choice(initializer_options) for i in range(n_layers_autoencoder)]
        bias_initializer_decoder = [np.random.choice(initializer_options) for i in range(n_layers_autoencoder)]
        bias_initializer_params_encoder = [randomize_params_for_tf_initializer(initializer) for initializer in
                                           bias_initializer_encoder]
        bias_initializer_params_decoder = [randomize_params_for_tf_initializer(initializer) for initializer in
                                           bias_initializer_decoder]

        # pick weight initializer randomly
        weights_initializer_encoder = [np.random.choice(initializer_options) for i in range(n_layers_autoencoder)]
        weights_initializer_decoder = [np.random.choice(initializer_options) for i in range(n_layers_autoencoder)]
        weights_initializer_params_encoder = [randomize_params_for_tf_initializer(initializer) for initializer in
                                           weights_initializer_encoder]
        weights_initializer_params_decoder = [randomize_params_for_tf_initializer(initializer) for initializer in
                                           weights_initializer_decoder]

        params["bias_initializer_encoder"] = bias_initializer_encoder
        params["bias_initializer_decoder"] = bias_initializer_decoder
        params["bias_initializer_params_encoder"] = bias_initializer_params_encoder
        params["bias_initializer_params_decoder"] = bias_initializer_params_decoder

        params["weights_initializer_encoder"] = weights_initializer_encoder
        params["weights_initializer_decoder"] = weights_initializer_decoder
        params["weights_initializer_params_encoder"] = weights_initializer_params_encoder
        params["weights_initializer_params_decoder"] = weights_initializer_params_decoder

        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
            params["input_dim_y"] = 1000
        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_intensities_sqrt" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_sqrt":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

        params["input_dim_x"] = 1
        params["n_classes"] = 2
        params["z_dim"] = 5

        params["selected_dataset"] = "mass_spec"

        params["only_train_autoencoder"] = True

        params["verbose"] = True
        # params["selected_autoencoder"] = "IncorporatingLabelInformation"
        # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        # update the parameters dependent on the network architecture
        # params = update_basic_network_params(params)

        params["AdamOptimizer_beta1_discriminator"] = 0.9
        params["AdamOptimizer_beta1_autoencoder"] = 0.9
        params["AdamOptimizer_beta1_generator"] = 0.9

        params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
        params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
        params['decaying_learning_rate_name_generator'] = "piecewise_constant"

        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                               "values": [0.0001, 1e-05, 1e-06]}

        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}
        params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                             "values": [0.1, 0.01, 0.001, 0.0001]}

        for j in range(1):
            aae = UnsupervisedAdversarialAutoencoder(params)
            aae.train(True)
            aae.reset_graph()


def param_search_batch_normalization():

    for i in range(1):

        params = get_default_parameters_mass_spec()

        params["summary_image_frequency"] = 100
        params["n_epochs"] = 101

        params["mz_loss_factor"] = 1
        params["intensity_loss_factor"] = 1

        # peak_encoding
        ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
         "only_intensities_distance", "raw", "raw_intensities_sqrt", "raw_sqrt"]

        # datasubset:
        ["identified", "unidentified", None]

        # smoothing_method
        ["loess", "gaussian_filter", "spline"]

        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "raw_sqrt",
                                               "use_smoothed_intensities": False, "data_subset": None,
                                               "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                               "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                               "include_molecular_weight_in_encoding": False,
                                               "include_charge_in_encoding": False,
                                               "smoothness_params": {"smoothing_method": "spline",
                                                                     "smoothness_frac": 0.0,
                                                                     "smoothness_spar": 0.1,
                                                                     "smoothness_sigma": 1}}

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [250, 250, 250]

        # activation functions
        batch_normalization_options = [None, "post_activation", "pre_activation"]

        n_layers_autoencoder = len(params["n_neurons_of_hidden_layer_x_autoencoder"]) + 1
        batch_normalization_encoder = [np.random.choice(batch_normalization_options) for i in range(n_layers_autoencoder)]
        batch_normalization_decoder = [np.random.choice(batch_normalization_options) for i in range(n_layers_autoencoder)]

        params["batch_normalization_encoder"] = batch_normalization_encoder
        params["batch_normalization_encoder"] = batch_normalization_decoder

        params["activation_function_encoder"] = ['relu', 'relu6', 'softsign', 'linear']
        params["activation_function_decoder"] = ['relu6', 'relu6', 'relu', 'sigmoid']

        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
            params["input_dim_y"] = 1000
        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_intensities_sqrt" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_sqrt":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

        params["input_dim_x"] = 1
        params["n_classes"] = 2
        params["z_dim"] = 15

        params["selected_dataset"] = "mass_spec"

        params["only_train_autoencoder"] = True

        params["verbose"] = True
        # params["selected_autoencoder"] = "IncorporatingLabelInformation"
        # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        # update the parameters dependent on the network architecture
        # params = update_basic_network_params(params)

        params["AdamOptimizer_beta1_discriminator"] = 0.9
        params["AdamOptimizer_beta1_autoencoder"] = 0.9
        params["AdamOptimizer_beta1_generator"] = 0.9

        params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
        params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
        params['decaying_learning_rate_name_generator'] = "piecewise_constant"

        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                               "values": [0.0001, 1e-05, 1e-06]}

        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}
        params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                             "values": [0.1, 0.01, 0.001, 0.0001]}

        for j in range(4):
            aae = UnsupervisedAdversarialAutoencoder(params)
            aae.train(True)
            aae.reset_graph()


def param_search_activation_functions():

    for i in range(1):

        params = get_default_parameters_mass_spec()

        params["summary_image_frequency"] = 100
        params["n_epochs"] = 101

        params["mz_loss_factor"] = 1
        params["intensity_loss_factor"] = 1

        # peak_encoding
        ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
         "only_intensities_distance", "raw", "raw_intensities_sqrt", "raw_sqrt"]

        # datasubset:
        ["identified", "unidentified", None]

        # smoothing_method
        ["loess", "gaussian_filter", "spline"]

        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "raw_sqrt",
                                               "use_smoothed_intensities": False, "data_subset": None,
                                               "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                               "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                               "include_molecular_weight_in_encoding": False,
                                               "include_charge_in_encoding": False,
                                               "smoothness_params": {"smoothing_method": "spline",
                                                                     "smoothness_frac": 0.0,
                                                                     "smoothness_spar": 0.1,
                                                                     "smoothness_sigma": 1}}

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [250, 250, 250]

        # activation functions
        activation_function_options = ["relu", "relu6", "elu", "softplus", "softsign", "sigmoid", "tanh",
                                       "leaky_relu", "linear"]

        n_layers_autoencoder = len(params["n_neurons_of_hidden_layer_x_autoencoder"]) + 1
        activation_function_encoder = [np.random.choice(activation_function_options) for i in range(n_layers_autoencoder)]
        activation_function_decoder = [np.random.choice(activation_function_options) for i in range(n_layers_autoencoder)]

        params["activation_function_encoder"] = activation_function_encoder
        params["activation_function_decoder"] = activation_function_decoder

        # params["activation_function_encoder"] = ['relu', 'relu6', 'softsign', 'linear']
        # params["activation_function_decoder"] = ['relu6', 'relu6', 'relu', 'sigmoid']

        # params["activation_function_encoder"] = ['relu', 'relu', 'relu', 'linear']
        # params["activation_function_decoder"] = ['sigmoid', 'relu', 'relu', 'linear']

        # params["activation_function_encoder"] = ['elu', 'relu', 'tanh', 'sigmoid']
        # params["activation_function_decoder"] = ['relu6', 'linear', 'tanh', 'softplus']

        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
            params["input_dim_y"] = 1000
        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_intensities_sqrt" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_sqrt":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

        params["input_dim_x"] = 1
        params["n_classes"] = 2
        params["z_dim"] = 15

        params["selected_dataset"] = "mass_spec"

        params["only_train_autoencoder"] = True

        params["verbose"] = True
        # params["selected_autoencoder"] = "IncorporatingLabelInformation"
        # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        # update the parameters dependent on the network architecture
        # params = update_basic_network_params(params)

        params["AdamOptimizer_beta1_discriminator"] = 0.9
        params["AdamOptimizer_beta1_autoencoder"] = 0.9
        params["AdamOptimizer_beta1_generator"] = 0.9

        params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
        params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
        params['decaying_learning_rate_name_generator'] = "piecewise_constant"

        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                               "values": [0.0001, 1e-05, 1e-06]}

        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}
        params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                             "values": [0.1, 0.01, 0.001, 0.0001]}

        for j in range(1):
            aae = UnsupervisedAdversarialAutoencoder(params)
            aae.train(True)
            aae.reset_graph()


def param_search_peak_encodings():

    for z_dim in [5]:
        for n_neurons_of_hidden_layer_x_autoencoder in [[250, 250, 250]]:
            #     for n_neurons_of_hidden_layer_x_autoencoder in [[2000, 2000, 2000, 2000], [1000, 1000], [1000, 1000, 1000],
            #                                                     [1000, 1000, 1000, 1000], [2000, 2000], [2000, 2000, 2000],
            #                                                 [5000, 5000], [500, 500, 500, 500, 500, 500], [125, 125, 125, 125, 125, 125]]:
            # for frac in [0.001, 0.005, 0.01, 0.05, 0.1]:
            # for peak_encoding in ["only_mz", "only_intensities", "distance", "location",
            #      "raw", "raw_intensities_sqrt", "raw_sqrt"]:
            for peak_encoding in ["binned"]:
                # for frac in [0.1]:

                params = get_default_parameters_mass_spec()
                params["summary_image_frequency"] = 100
                params["n_epochs"] = 101

                params["mz_loss_factor"] = 1
                params["intensity_loss_factor"] = 1

                # peak_encoding
                ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
                 "only_intensities_distance", "raw", "raw_intensities_sqrt", "raw_sqrt"]

                # datasubset:
                ["identified", "unidentified", None]

                # smoothing_method
                ["loess", "gaussian_filter", "spline"]

                params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": peak_encoding,
                                                       "use_smoothed_intensities": False, "data_subset": None,
                                                       "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                                       "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                                       "include_molecular_weight_in_encoding": False,
                                                       "include_charge_in_encoding": False,
                                                       "smoothness_params": {"smoothing_method": None,
                                                                             "smoothness_frac": 0.0,
                                                                             "smoothness_spar": 0.0,
                                                                             "smoothness_sigma": 0.0}}

                params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
                    [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                     params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
                if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
                    params["input_dim_y"] = 1000
                elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                                params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                                params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                                params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
                    params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
                elif params["mass_spec_data_properties"]["peak_encoding"] == "raw" or \
                                params["mass_spec_data_properties"]["peak_encoding"] == "raw_intensities_sqrt" or \
                                params["mass_spec_data_properties"]["peak_encoding"] == "raw_sqrt":
                    params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                        [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                         params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

                params["input_dim_x"] = 1
                params["n_classes"] = 2
                params["z_dim"] = z_dim

                params["selected_dataset"] = "mass_spec"

                params["only_train_autoencoder"] = True

                params["verbose"] = True
                # params["selected_autoencoder"] = "IncorporatingLabelInformation"
                # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
                params["selected_autoencoder"] = "Unsupervised"
                params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

                params["n_neurons_of_hidden_layer_x_autoencoder"] = n_neurons_of_hidden_layer_x_autoencoder

                # update the parameters dependent on the network architecture
                params = update_basic_network_params(params)

                params["AdamOptimizer_beta1_discriminator"] = 0.9
                params["AdamOptimizer_beta1_autoencoder"] = 0.9
                params["AdamOptimizer_beta1_generator"] = 0.9

                params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
                params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
                params['decaying_learning_rate_name_generator'] = "piecewise_constant"

                params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                                       "values": [0.0001, 1e-05, 1e-06]}

                params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                         "values": [0.1, 0.01, 0.001, 0.0001]}
                params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                                     "values": [0.1, 0.01, 0.001, 0.0001]}

                # for j in range(10):
                for j in range(5):
                    aae = UnsupervisedAdversarialAutoencoder(params)
                    # aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

                    aae.train(True)
                    aae.reset_graph()


def param_search_smoothing_intensities():

        #
        # for z_dim in [2, 5, 10, 15, 35, 65, 90, 120, 150, 200]:
        # for z_dim in [15, 50, 2]:
        for z_dim in [5]:
            for n_neurons_of_hidden_layer_x_autoencoder in [[250, 250, 250]]:
        #     for n_neurons_of_hidden_layer_x_autoencoder in [[2000, 2000, 2000, 2000], [1000, 1000], [1000, 1000, 1000],
        #                                                     [1000, 1000, 1000, 1000], [2000, 2000], [2000, 2000, 2000],
        #                                                 [5000, 5000], [500, 500, 500, 500, 500, 500], [125, 125, 125, 125, 125, 125]]:
                # for frac in [0.001, 0.005, 0.01, 0.05, 0.1]:
                for frac in [0.001, 0.1, 0.3, 0.5, 1.0]:
                # for frac in [0.1]:
                # for frac in [0.1]:
                    for smoothing_method in ["loess"]:

                        params = get_default_parameters_mass_spec()
                        params["summary_image_frequency"] = 100
                        params["n_epochs"] = 101

                        params["mz_loss_factor"] = 1
                        params["intensity_loss_factor"] = 1

                        # peak_encoding
                        ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
                         "only_intensities_distance", "raw", "raw_intensities_sqrt", "raw_sqrt"]

                        # datasubset:
                        ["identified", "unidentified", None]

                        # smoothing_method
                        ["loess", "gaussian_filter", "spline"]

                        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "only_intensities",
                                                               "use_smoothed_intensities": True, "data_subset": None,
                                                               "n_peaks_to_keep": 50, "max_intensity_value": 5000,
                                                               "max_mz_value": 5000, "charge": None, "normalize_data": True,
                                                               "include_molecular_weight_in_encoding": False,
                                                               "include_charge_in_encoding": False,
                                                               "smoothness_params": {"smoothing_method": smoothing_method,
                                                                                     "smoothness_frac": frac,
                                                                                     "smoothness_spar": frac,
                                                                                     "smoothness_sigma": frac}}

                        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
                            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
                        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
                            params["input_dim_y"] = 1000
                        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
                            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
                        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw" or \
                                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_intensities_sqrt" or \
                                        params["mass_spec_data_properties"]["peak_encoding"] == "raw_sqrt":
                            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
                                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

                        params["input_dim_x"] = 1
                        params["n_classes"] = 2
                        params["z_dim"] = z_dim

                        params["selected_dataset"] = "mass_spec"

                        params["only_train_autoencoder"] = True

                        params["verbose"] = True
                        # params["selected_autoencoder"] = "IncorporatingLabelInformation"
                        # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
                        params["selected_autoencoder"] = "Unsupervised"
                        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

                        params["n_neurons_of_hidden_layer_x_autoencoder"] = n_neurons_of_hidden_layer_x_autoencoder

                        # update the parameters dependent on the network architecture
                        params = update_basic_network_params(params)

                        params["AdamOptimizer_beta1_discriminator"] = 0.9
                        params["AdamOptimizer_beta1_autoencoder"] = 0.9
                        params["AdamOptimizer_beta1_generator"] = 0.9

                        params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
                        params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
                        params['decaying_learning_rate_name_generator'] = "piecewise_constant"

                        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                                               "values": [0.0001, 1e-05, 1e-06]}

                        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}
                        params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                                             "values": [0.1, 0.01, 0.001, 0.0001]}

                        # for j in range(10):
                        for j in range(5):

                            aae = UnsupervisedAdversarialAutoencoder(params)
                            # aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

                            aae.train(True)
                            aae.reset_graph()


def param_search_only_mz():

    # for i in [2, 5, 10, 15, 35, 65, 90, 120, 150, 200]:
    for i in [2]:

        params = get_default_parameters_mass_spec()

        params["summary_image_frequency"] = 5
        params["n_epochs"] = 101

        # peak_encoding
        ["only_mz", "only_intensities", "only_mz_charge_label", "distance", "location", "binned",
         "only_intensities_distance", "raw"]

        # datasubset:
        ["identified", "unidentified", None]

        # params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "only_intensities",
        #                                        "include_charge_in_encoding": False,
        #                                        "include_molecular_weight_in_encoding": False, "charge": "2",
        #                                        "normalize_data": False, "n_peaks_to_keep": 50,
        #                                        "max_intensity_value": 5000}

        # smoothing_method
        ["lowess", "gaussian_filter"]

        params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "only_intensities",
                                               "use_smoothed_intensities": True, "data_subset": None,
                                               "n_peaks_to_keep": 50, "max_intensity_value": 2000,
                                               "max_mz_value": 2000, "charge": None, "normalize_data": False,
                                               "include_molecular_weight_in_encoding": False,
                                               "include_charge_in_encoding": False,
                                               "smoothness_params": {"smoothing_method": "lowess",
                                                                     "smoothness_frac": 0.3,
                                                                     "smoothness_sigma": 1,
                                                                     "smoothing_n_gaussians": 15}}

        params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
        if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
            params["input_dim_y"] = 1000
        elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_mz_charge_label" or \
                        params["mass_spec_data_properties"]["peak_encoding"] == "only_intensities_distance":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]
        elif params["mass_spec_data_properties"]["peak_encoding"] == "raw":
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 2 + sum(
            [params["mass_spec_data_properties"]["include_charge_in_encoding"],
             params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])

        params["input_dim_x"] = 1
        params["n_classes"] = 2
        params["z_dim"] = i

        params["selected_dataset"] = "mass_spec"

        params["only_train_autoencoder"] = True

        params["verbose"] = True
        # params["selected_autoencoder"] = "IncorporatingLabelInformation"
        # params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [1000, 1000, 1000]

        params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        params["batch_normalization_encoder"] = [None] * 3
        params["batch_normalization_decoder"] = [None] * 3

        params["AdamOptimizer_beta1_discriminator"] = 0.9
        params["AdamOptimizer_beta1_autoencoder"] = 0.9
        params["AdamOptimizer_beta1_generator"] = 0.9

        params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
        params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
        params['decaying_learning_rate_name_generator'] = "piecewise_constant"


        # params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [250, 1500],
        #                                                        "values": [0.01, 0.001, 0.000001]}

        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [500, 1500],
                                                               "values": [0.0001, 1e-05, 1e-06]}

        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}
        params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                             "values": [0.1, 0.01, 0.001, 0.0001]}

        # aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSameTopology(params)
        # aae = LearningPriorsAdversarialAutoencoder(params)

        # for j in range(10):
        for j in range(1):

            aae = UnsupervisedAdversarialAutoencoder(params)
            # aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

            aae.train(True)
            aae.reset_graph()


def update_basic_network_params(params):
    n_autoencoder_layers = len(params["n_neurons_of_hidden_layer_x_autoencoder"]) + 1
    n_discriminator_layers = len(params["n_neurons_of_hidden_layer_x_discriminator"]) + 1
    params["batch_normalization_encoder"] = [None] * n_autoencoder_layers
    params["batch_normalization_decoder"] = [None] * n_autoencoder_layers
    params["batch_normalization_discriminator"] = [None] * n_discriminator_layers
    params["bias_initializer_encoder"] = ["zeros_initializer"] * n_autoencoder_layers
    params["bias_initializer_decoder"] = ["zeros_initializer"] * n_autoencoder_layers
    params["bias_initializer_discriminator"] = ["zeros_initializer"] * n_discriminator_layers
    params["bias_initializer_params_encoder"] = [{}] * n_autoencoder_layers
    params["bias_initializer_params_decoder"] = [{}] * n_autoencoder_layers
    params["bias_initializer_params_discriminator"] = [{}] * n_discriminator_layers
    params["dropout_encoder"] = [0.0] * n_autoencoder_layers
    params["dropout_decoder"] = [0.0] * n_autoencoder_layers
    params["dropout_discriminator"] = [0.0] * n_discriminator_layers
    params["weights_initializer_encoder"] = ["truncated_normal_initializer"] * n_autoencoder_layers
    params["weights_initializer_decoder"] = ["truncated_normal_initializer"] * n_autoencoder_layers
    params["weights_initializer_discriminator"] = ["truncated_normal_initializer"] * n_discriminator_layers
    params["weights_initializer_params_encoder"] = [{"mean": 0, "stddev": 0.1}] * n_autoencoder_layers
    params["weights_initializer_params_decoder"] = [{"mean": 0, "stddev": 0.1}] * n_autoencoder_layers
    params["weights_initializer_params_discriminator"] = [{"mean": 0, "stddev": 0.1}] * n_discriminator_layers

    # TODO:
    # params["activation_function_encoder"] = ['relu'] * (n_autoencoder_layers-1) + ['linear']
    # params["activation_function_decoder"] = ['sigmoid'] + ['relu'] * (n_autoencoder_layers-2) + ['linear']
    # params["activation_function_discriminator"] = ['relu'] * (n_discriminator_layers-1) + ['linear']

    return params


def testing():
    """
    :return:
    """

    selected_datasets = ["MNIST", "SVHN", "cifar10", "custom"]
    selected_autoencoders = ["Unsupervised", "Supervised", "SemiSupervised"]
    decaying_learning_rate_names = ["exponential_decay", "inverse_time_decay", "natural_exp_decay",
                                    "piecewise_constant", "polynomial_decay", "static"]
    activation_functions = ["relu", "relu6", "crelu", "elu", "softplus", "softsign", "sigmoid", "tanh", "leaky_relu",
                            "linear"]
    weight_bias_initializers = ["constant_initializer", "random_normal_initializer", "truncated_normal_initializer",
                                "random_uniform_initializer", "uniform_unit_scaling_initializer", "zeros_initializer",
                                "ones_initializer", "orthogonal_initializer"]
    batch_normalization = ["post_activation", "pre_activation", None]

    # params:
    #   constant_initializer: initial value
    #   random_normal_initializer: mean, stddev
    #   truncated_normal_initializer: mean, stddev
    #   random_uniform_initializer: minval, maxval
    #   uniform_unit_scaling_initializer: factor: Float. A multiplicative factor by which the values will be scaled.
    #   orthogonal_initializer: gain: Float. Multiplicative factor to apply to the orthogonal matrix

    if False:
        print("training only mz values")
        params = get_params_from_params_file("D:/Results/mass_spec_data/only_mz_values/varying_z_dim_large_scale/network_architecture/2018-07-12_17_01_28_mass_spec/log/params.txt")
        params["mass_spec_data_properties"]["max_mz_value"] = 5000
        params["mass_spec_data_properties"]["charge"] = None
        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.train(True)
        return

    if False:

        # incorporating label information
        default_params = get_default_parameters_svhn()
        params = get_params_from_params_file("D:/Results/IncorporatingLabelInformation/2018-03-02_15_49_50_SVHN/log/params.txt")

        params = default_params

        params["z_dim"] = 12

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [1000, 1000]

        params["batch_normalization_encoder"] = [None, None, None]
        params["batch_normalization_decoder"] = [None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None]

        params["activation_function_encoder"] = ['relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'sigmoid']
        params["activation_function_discriminator"] = ['relu', 'relu', 'linear']

        params["decaying_learning_rate_name_autoencoder"] = "piecewise_constant"
        params["decaying_learning_rate_name_discriminator"] = "piecewise_constant"
        params["decaying_learning_rate_name_generator"] = "piecewise_constant"

        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [250], "values": [0.0001, 0.00001]}
        params["decaying_learning_rate_params_discriminator"] = {"boundaries": [250], "values": [0.0001, 0.00001]}
        params["decaying_learning_rate_params_generator"] = {"boundaries": [250], "values": [0.0001, 0.00001]}

        params["AdamOptimizer_beta1_autoencoder"] = 0.5
        params["AdamOptimizer_beta1_discriminator"] = 0.5

        params["loss_function_discriminator"] = "sigmoid_cross_entropy"
        params["loss_function_generator"] = "sigmoid_cross_entropy"

        params["results_path"] = "../../results/IncorporatingLabelInformation"
        params["n_epochs"] = 1001
        params["summary_image_frequency"] = 50
        aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
        aae.train(True)
        aae.reset_graph()

        return


        default_params = get_default_parameters_svhn()
        params = get_params_from_params_file("D:/Results/Supervised/svhn_varying_z_dim/varying_lr/2018-05-23_21_35_50_SVHN_good/log/params.txt")
        # for key, value in params.items():
        #     default_params[key] = value
        #
        # params = update_basic_network_params(default_params)
        #
        # for key, value in params.items():
        #     default_params[key] = value

        params["results_path"] = "../../results/Supervised"
        params["n_epochs"] = 101
        params["summary_image_frequency"] = 5
        aae = SupervisedAdversarialAutoencoder(params)
        aae.train(True)
        aae.reset_graph()


        return

        print("training svhn")
        params = get_default_parameters_svhn()
        params["n_epochs"] = 101
        params["summary_image_frequency"] = 10
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")
        params["save_final_model"] = True
        params["write_tensorboard"] = True
        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.train(True)
        aae.reset_graph()
        return

    if False:

        # default_params = get_default_parameters_mnist()
        # params = get_params_from_params_file("D:/Results/interesting_results/older_results/Unsupervised_MNIST_image_grid_weights/log/params.txt")
        # for key, value in params.items():
        #     default_params[key] = value
        #
        # params = update_basic_network_params(default_params)
        #
        # for key, value in params.items():
        #     default_params[key] = value
        #
        # default_params["results_path"] = "../../results/Supervised"
        # default_params["n_epochs"] = 101
        # default_params["summary_image_frequency"] = 5
        # aae = SupervisedAdversarialAutoencoder(default_params)
        # aae.train(True)
        # aae.reset_graph()

        # default params
        if False:
            print("training MNIST")
            params = get_default_parameters_mnist()
            params["n_epochs"] = 101
            params["summary_image_frequency"] = 5
            params["z_dim"] = 2
            params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")
            params["save_final_model"] = True
            params["write_tensorboard"] = True
            aae = UnsupervisedAdversarialAutoencoder(params)
            aae.train(True)
            aae.reset_graph()

        # incorporate label information
        if True:
            print("training MNIST")
            params = get_default_parameters_mnist()
            params["n_epochs"] = 201
            params["summary_image_frequency"] = 10
            params["z_dim"] = 2
            params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
            params["save_final_model"] = True
            params["write_tensorboard"] = True
            params["AdamOptimizer_beta1_autoencoder"] = 0.5
            aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
            aae.train(True)
            aae.reset_graph()


        return

    if False:
        # aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Desktop\\interesting_results\\older_results\\2018-03-02_15_49_50_SVHN\\log\\params.txt", "Supervised")
        # aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Desktop\\interesting_results\\older_results\\2018-03-02_15_49_50_SVHN\\log\\params_activation_functions_modified.txt", "Supervised")
        # aae.train(True)
        aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Dropbox\\Studium\\Studium\\Master\\4. Semester\\Masterarbeit\\AdversarialAutoencoder\\results\\Unsupervised\\2018-07-10_12_59_57_mass_spec\\log\\params.txt", "Unsupervised")
        aae.train(True)
        return

    if False:
        params = get_default_parameters("Unsupervised", "MNIST")
        params["n_epochs"] = 21
        params["save_final_model"] = True
        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.train(True)
        return

    # try reloading a saved autoencoder
    if False:
        params = get_params_from_params_file("C:/Users/Telcontar/Dropbox/Studium/Studium/Master/4. Semester/Masterarbeit/AdversarialAutoencoder/results/Unsupervised/2018-08-02_17_48_33_MNIST/log/params.txt")
        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.set_result_folder_name("/2018-08-02_17_48_33_MNIST")
        aae.train(False)
        return

    if False:
        params = get_params_from_params_file("D:/Results/mass_spec_data/only_intensities/smoothing_desktop/2018-08-13_16_56_02_mass_spec/log/params.txt")
        params["n_epochs"] = 1001
        params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [6500, 9500],
                                                               "values": [0.0001, 1e-05, 1e-06]}
        params["mass_spec_data_properties"]["normalize_data"] = True

        aae = UnsupervisedAdversarialAutoencoder(params)
        aae.train(True)
        return

    if False:
        try_mass_spec_parameter_combinations()
        return

    if True:
        # param_search_incorporating_label_information()

        if True:
            param_search_weight_bias_initializers()
            return

        # for cluster:
        if True:
            param_search_activation_functions()
            return

        if False:
            param_search_peak_encodings()
            return


        param_search_smoothing_intensities()
        # param_search_only_mz()
        # param_search_mass_spec_data()

        return

        # TODO: batch_size of 101 is not working

        # do_randomsearch(1, "activation_function_encoder", batch_size=103, n_epochs=2, summary_image_frequency=5,
        #                 learning_rate_autoencoder=[np.random.uniform(0.2, 0.001)*2])
        #
        do_randomsearch(100)
        # do_randomsearch(2, "batch_size", "activation_function_encoder", n_epochs=5, summary_image_frequency=5,
        #                 learning_rate_autoencoder=[np.random.uniform(0.2, 0.001)*2])
        #
        # return

        """
        test grid search
        """
        do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="MNIST", n_epochs=[2],
                      n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                      n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]],
                      decaying_learning_rate_params_autoencoder=[{'learning_rate': 0.0001}]
        )

        return

        do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="MNIST", n_epochs=[2],
                      n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                      n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]],
                      weights_initializer_params_discriminator=[
                          [{"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                           {"mean": 0, "stddev": 0.1}],
                      [{"mean": 0, "stddev": 0.12345}, {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
                       {"mean": 0, "stddev": 0.1}],
        [{"mean": 0, "stddev": 0.67890}, {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
         {"mean": 0, "stddev": 0.1}],
        [{"mean": 0, "stddev": 0.13579}, {"mean": 0, "stddev": 0.1}, {"mean": 0, "stddev": 0.1},
        {"mean": 0, "stddev": 0.1}]]
        )

        return

    """
    test yeast mass spec data:
    """
    if False:

        for i in [2, 5, 10, 15, 35, 65, 90, 120, 150, 200]:

            params = get_default_parameters_mass_spec()

            params["summary_image_frequency"] = 25

            params["n_epochs"] = 501

            params["mass_spec_data_properties"] = {"organism_name": "yeast", "peak_encoding": "only_mz",
                                                   "include_charge_in_encoding": False, "use_smoothed_intensities": True,
                                                   "smoothness_sigma": 1,
                                                   "include_molecular_weight_in_encoding": False, "charge": "2",
                                                   "normalize_data": False, "n_peaks_to_keep": 50,
                                                   "max_intensity_value": 5000}
            params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"] * 3 + sum(
                [params["mass_spec_data_properties"]["include_charge_in_encoding"],
                 params["mass_spec_data_properties"]["include_molecular_weight_in_encoding"]])
            if params["mass_spec_data_properties"]["peak_encoding"] == "binned":
                params["input_dim_y"] = 1000
            elif params["mass_spec_data_properties"]["peak_encoding"] == "only_mz":
                params["input_dim_y"] = params["mass_spec_data_properties"]["n_peaks_to_keep"]

            params["input_dim_x"] = 1
            params["n_classes"] = 2
            params["z_dim"] = i

            params["selected_dataset"] = "mass_spec"

            params["only_train_autoencoder"] = True

            params["verbose"] = True
            params["selected_autoencoder"] = "Unsupervised"
            params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

            params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000, 1000, 1000]
            params["n_neurons_of_hidden_layer_x_discriminator"] = [1000, 1000, 1000]

            params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0]
            params["batch_normalization_encoder"] = [None]*5
            params["batch_normalization_decoder"] = [None]*5

            params["AdamOptimizer_beta1_discriminator"] = 0.5
            params["AdamOptimizer_beta1_autoencoder"] = 0.5
            params["AdamOptimizer_beta1_generator"] = 0.5

            params['decaying_learning_rate_name_autoencoder'] = "piecewise_constant"
            params['decaying_learning_rate_name_discriminator'] = "piecewise_constant"
            params['decaying_learning_rate_name_generator'] = "piecewise_constant"

            params["decaying_learning_rate_params_autoencoder"] = {"boundaries": [250, 1500],
                                                                   "values": [0.01, 0.001, 0.000001]}
            params["decaying_learning_rate_params_discriminator"] = {"boundaries": [500, 1500, 2500],
                                                                     "values": [0.1, 0.01, 0.001, 0.0001]}
            params["decaying_learning_rate_params_generator"] = {"boundaries": [500, 1500, 2500],
                                                                 "values": [0.1, 0.01, 0.001, 0.0001]}

            # aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
            # aae = LearningPriorsAdversarialAutoencoderSupervised(params)
            # aae = LearningPriorsAdversarialAutoencoderSameTopology(params)
            # aae = LearningPriorsAdversarialAutoencoder(params)
            aae = UnsupervisedAdversarialAutoencoder(params)

            aae.train(True)
            aae.reset_graph()

        return

    if False:
        # code for the cluster
        for i in range(2, 21):

            params = get_default_parameters_mnist()
            params["selected_dataset"] = "MNIST"

            params["z_dim"] = i

            params["verbose"] = True
            params["selected_autoencoder"] = "Supervised"
            params["results_path"] = get_result_path_for_selected_autoencoder("Supervised")
            params["summary_image_frequency"] = 5
            params["n_epochs"] = 16
            params["batch_normalization_encoder"] = [None, None, None, None, None, None, None]
            params["batch_normalization_decoder"] = [None, None, None, None, None, None, None]
            params["batch_normalization_discriminator"] = [None, None, None, None, None, None, None]

            params["n_neurons_of_hidden_layer_x_autoencoder"] = [3000, 2000, 1000, 500, 250, 125]
            params["n_neurons_of_hidden_layer_x_discriminator"] = [3000, 2000, 1000, 500, 250, 125]

            params["activation_function_encoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
            params["activation_function_decoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
            params["activation_function_discriminator"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']

            params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            params["dropout_decoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            params["dropout_discriminator"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
            params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
            params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

            params['decaying_learning_rate_name_autoencoder'] = "static"
            params['decaying_learning_rate_name_discriminator'] = "static"
            params['decaying_learning_rate_name_generator'] = "static"

            params['bias_initializer_encoder'] = ["zeros_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['bias_initializer_decoder'] = ["zeros_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['bias_initializer_discriminator'] = ["zeros_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_discriminator"])

            params['bias_initializer_params_encoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['bias_initializer_params_decoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['bias_initializer_params_discriminator'] = [{}] * len(
                params["n_neurons_of_hidden_layer_x_discriminator"])

            params['weights_initializer_encoder'] = ["truncated_normal_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['weights_initializer_decoder'] = ["truncated_normal_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['weights_initializer_discriminator'] = ["truncated_normal_initializer"] * len(
                params["n_neurons_of_hidden_layer_x_discriminator"])

            params['weights_initializer_params_encoder'] = [{"mean": 0, "stddev": 0.1}] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['weights_initializer_params_decoder'] = [{"mean": 0, "stddev": 0.1}] * len(
                params["n_neurons_of_hidden_layer_x_autoencoder"])
            params['weights_initializer_params_discriminator'] = [{"mean": 0, "stddev": 0.1}] * len(
                params["n_neurons_of_hidden_layer_x_discriminator"])

            aae = SupervisedAdversarialAutoencoder(params)
            # aae = UnsupervisedAdversarialAutoencoder(params)
            # aae = SemiSupervisedAdversarialAutoencoder(params)

            aae.train(True)
            aae.reset_graph()
        return

    if False:

        params = get_default_parameters_mnist()
        params["selected_dataset"] = "MNIST"

        params["z_dim"] = 150

        params["verbose"] = True
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")
        params["summary_image_frequency"] = 50
        params["n_labeled"] = 10000
        params["n_epochs"] = 3
        params["batch_normalization_encoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None, None, None]

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [3000, 2000, 1000, 500, 250, 125]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [3000, 2000, 1000, 500, 250, 125]

        params["activation_function_encoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        params["activation_function_discriminator"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']

        params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_discriminator"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

        params['decaying_learning_rate_name_autoencoder'] = "static"
        params['decaying_learning_rate_name_discriminator'] = "static"
        params['decaying_learning_rate_name_generator'] = "static"

        params['bias_initializer_encoder'] = ["zeros_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_decoder'] = ["zeros_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_discriminator'] = ["zeros_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_discriminator"])

        params['bias_initializer_params_encoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_decoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_discriminator'] = [{}] * len(
            params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_encoder'] = ["truncated_normal_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_decoder'] = ["truncated_normal_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_discriminator'] = ["truncated_normal_initializer"] * len(
            params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_params_encoder'] = [{"mean": 0, "stddev": 0.1}] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_decoder'] = [{"mean": 0, "stddev": 0.1}] * len(
            params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_discriminator'] = [{"mean": 0, "stddev": 0.1}] * len(
            params["n_neurons_of_hidden_layer_x_discriminator"])

        # aae = SupervisedAdversarialAutoencoder(params)
        aae = UnsupervisedAdversarialAutoencoder(params)
        # aae = UnsupervisedClusteringAdversarialAutoencoder(params)
        # aae = SemiSupervisedAdversarialAutoencoder(params)
        # aae = IncorporatingLabelInformationAdversarialAutoencoder(params)
        # aae = UnsupervisedClusteringAdversarialAutoencoder(params)

        aae.train(True)
        aae.reset_graph()

        return

    if False:
        params = get_default_parameters_mnist()
        params["selected_dataset"] = "MNIST"

        params["z_dim"] = 4
        params["verbose"] = True
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")
        params["summary_image_frequency"] = 10
        params["n_epochs"] = 101
        params["batch_normalization_encoder"] = [None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None]

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator_c"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator_g"] = [1000, 1000]

        params["activation_function_encoder"] = ['relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'sigmoid']
        params["activation_function_discriminator_c"] = ['relu', 'relu', 'linear']
        params["activation_function_discriminator_g"] = ['relu', 'relu', 'linear']

        params["dropout_encoder"] = [0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0]
        params["dropout_discriminator_c"] = [0.0, 0.0, 0.0]
        params["dropout_discriminator_g"] = [0.0, 0.0, 0.0]

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_discriminator_gaussian"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_discriminator_categorical"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_supervised_encoder"] = {"learning_rate": 0.001}

        # learning_priors_aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
        # learning_priors_aae = LearningPriorsAdversarialAutoencoderSupervised(params)
        # unsupervised_clustering_aae = UnsupervisedClusteringAdversarialAutoencoder(params)
        # unsupervised_clustering_aae = SemiSupervisedAdversarialAutoencoder(params)
        aae = UnsupervisedAdversarialAutoencoder(params)

        aae.train(True)

        return

    if False:

        params = get_default_parameters_svhn()
        params["selected_dataset"] = "SVHN"

        params["z_dim"] = 2

        params["verbose"] = True
        params["selected_autoencoder"] = "IncorporatingLabelInformation"
        params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["summary_image_frequency"] = 5
        params["n_epochs"] = 101
        params["batch_normalization_encoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None, None, None]

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [3000, 2000, 1000, 500, 250, 125]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [3000, 2000, 1000, 500, 250, 125]

        params["activation_function_encoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        params["activation_function_discriminator"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']

        params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_discriminator"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

        params['decaying_learning_rate_name_autoencoder'] = "static"
        params['decaying_learning_rate_name_discriminator'] = "static"
        params['decaying_learning_rate_name_generator'] = "static"

        params['bias_initializer_encoder'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_decoder'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_discriminator'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['bias_initializer_params_encoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_decoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_discriminator'] = [{}] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_encoder'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_decoder'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_discriminator'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_params_encoder'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_decoder'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_discriminator'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

        aae.train(True)

        return

    if False:
        params = get_default_parameters_mnist()
        params["selected_dataset"] = "MNIST"

        params["z_dim"] = 2
        params["verbose"] = True
        params["selected_autoencoder"] = "IncorporatingLabelInformation"
        params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["summary_image_frequency"] = 10
        params["n_epochs"] = 101
        params["batch_normalization_encoder"] = [None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None]

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator_c"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator_g"] = [1000, 1000]

        params["activation_function_encoder"] = ['relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'sigmoid']
        params["activation_function_discriminator_c"] = ['relu', 'relu', 'linear']
        params["activation_function_discriminator_g"] = ['relu', 'relu', 'linear']

        params["dropout_encoder"] = [0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0]
        params["dropout_discriminator_c"] = [0.0, 0.0, 0.0]
        params["dropout_discriminator_g"] = [0.0, 0.0, 0.0]

        params['decaying_learning_rate_name_autoencoder'] = "static"
        params['decaying_learning_rate_name_discriminator'] = "static"
        params['decaying_learning_rate_name_generator'] = "static"

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.001}
        params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.001}

        params["AdamOptimizer_beta1_autoencoder"] = 0.5
        params["AdamOptimizer_beta1_discriminator"] = 0.5
        params["AdamOptimizer_beta1_generator"] = 0.5

        # aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSupervised(params)
        # aae = UnsupervisedClusteringAdversarialAutoencoder(params)
        # aae = SemiSupervisedAdversarialAutoencoder(params)
        aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

        aae.train(True)

        return

    if False:
        params = get_default_parameters_svhn()
        params["selected_dataset"] = "SVHN"

        params["z_dim"] = 2

        params["verbose"] = True
        params["selected_autoencoder"] = "IncorporatingLabelInformation"
        params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["summary_image_frequency"] = 5
        params["n_epochs"] = 101
        params["batch_normalization_encoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None, None, None]

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [3000, 2000, 1000, 500, 250, 125]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [3000, 2000, 1000, 500, 250, 125]

        params["activation_function_encoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        params["activation_function_discriminator"] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']

        params["dropout_encoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["dropout_discriminator"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

        params['decaying_learning_rate_name_autoencoder'] = "static"
        params['decaying_learning_rate_name_discriminator'] = "static"
        params['decaying_learning_rate_name_generator'] = "static"

        params['bias_initializer_encoder'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_decoder'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_discriminator'] = ["zeros_initializer"] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['bias_initializer_params_encoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_decoder'] = [{}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['bias_initializer_params_discriminator'] = [{}] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_encoder'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_decoder'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_discriminator'] = ["truncated_normal_initializer"] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        params['weights_initializer_params_encoder'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_decoder'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_autoencoder"])
        params['weights_initializer_params_discriminator'] = [{"mean": 0, "stddev": 0.1}] * len(params["n_neurons_of_hidden_layer_x_discriminator"])

        # aae = SupervisedAdversarialAutoencoder(params)
        # aae = UnsupervisedAdversarialAutoencoder(params)
        # aae = DimensionalityReductionAdversarialAutoencoder(params)
        # aae = SemiSupervisedAdversarialAutoencoder(params)
        aae = IncorporatingLabelInformationAdversarialAutoencoder(params)

        aae.train(True)

        return

    if False:

        # params = get_default_parameters_svhn()
        # params["selected_dataset"] = "SVHN"

        params = get_default_parameters_mnist()
        params["selected_dataset"] = "MNIST"

        params["verbose"] = True
        params["selected_autoencoder"] = "Supervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Supervised")

        # aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSameTopology(params)
        # aae = LearningPriorsAdversarialAutoencoder(params)
        aae = SupervisedAdversarialAutoencoder(params)

        aae.train(True)

        return

    # do_randomsearch(1, selected_autoencoder="Unsupervised", selected_dataset="MNIST", n_epochs=101, verbose=True,
    #                 batch_normalization_encoder=[None]*5,
    #                 batch_normalization_decoder=[None]*5,
    #                 dropout_encoder=[0.0, 0.0, 0.0, 0.0, 0.0])
    #
    # return

    # do_randomsearch(1, selected_autoencoder="Supervised", selected_dataset="SVHN", n_epochs=1, verbose=True,
    #                 batch_normalization_encoder=[None]*5,
    #                 batch_normalization_decoder=[None]*5,
    #                 dropout_encoder=[0.0, 0.0, 0.0, 0.0, 0.0]
    #                 )
    #
    # return

    # do_randomsearch(1, selected_autoencoder="Supervised", selected_dataset="SVHN", n_epochs=201, verbose=True,
    #                 z_dim=8, batch_size=100, save_final_model=True,
    #                 summary_image_frequency=5)
    #
    # return


    # SVHN paper parameters
    do_randomsearch(1, selected_autoencoder="SemiSupervised", selected_dataset="SVHN", n_epochs=501, verbose=True,
                    z_dim=20, batch_size=100, save_final_model=True,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 3000],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 3000],
                    dropout_encoder=[0.2, 0.0, 0.0, 0.0, 0.0],
                    summary_image_frequency=50,
                    # batch_normalization_encoder=["post_activation", "post_activation", "post_activation"],
                    # batch_normalization_decoder=["post_activation", "post_activation", "post_activation"],
                    batch_normalization_encoder=[None, None, None],
                    batch_normalization_decoder=[None, None, None],
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    activation_function_encoder=['relu', 'relu', 'linear'],
                    activation_function_decoder=['sigmoid', 'relu', 'sigmoid'],
                    activation_function_discriminator=['relu', 'relu', 'linear'],
                    decaying_learning_rate_name_autoencoder="piecewise_constant",
                    decaying_learning_rate_name_discriminator="piecewise_constant",
                    decaying_learning_rate_name_generator="piecewise_constant",
                    decaying_learning_rate_name_supervised_encoder="piecewise_constant",
                    decaying_learning_rate_params_autoencoder={"boundaries": [250], "values": [0.0001, 0.00001]},
                    decaying_learning_rate_params_discriminator={"boundaries": [250], "values": [0.01, 0.001]},
                    decaying_learning_rate_params_generator={"boundaries": [250], "values": [0.01, 0.001]},
                    decaying_learning_rate_params_supervised_encoder={"boundaries": [250], "values": [0.1, 0.01]}
    )
    return

    # aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Desktop\\interesting_results\\older_results\\2018-03-02_15_49_50_SVHN\\log\\params.txt", "Supervised")
    # aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Desktop\\interesting_results\\older_results\\2018-03-02_15_49_50_SVHN\\log\\params_activation_functions_modified.txt", "Supervised")
    # aae.train(True)

    do_randomsearch(1, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=51, verbose=True,
                    z_dim=2, batch_size=100, save_final_model=True,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    # activation_function_encoder=['relu']*4,
                    # activation_function_decoder='relu',
                    # activation_function_discriminator='relu',
                    decaying_learning_rate_name_autoencoder="static",
                    decaying_learning_rate_name_discriminator="static",
                    decaying_learning_rate_name_generator="static",
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 3000],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 3000],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0],
                    activation_function_encoder=['relu', 'relu', 'linear'],
                    activation_function_decoder=['sigmoid', 'relu', 'sigmoid'],
                    activation_function_discriminator=['relu', 'relu', 'linear']
                    # n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    # n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    # bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    # bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

    return

    # try overfitting

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="cifar10", n_epochs=10000, verbose=True,
                    z_dim=32,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    decaying_learning_rate_name_autoencoder=None,
                    decaying_learning_rate_name_discriminator=None,
                    decaying_learning_rate_name_generator=None,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 3000, 3000],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 3000, 3000],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0]
                    )

    return

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="SVHN", n_epochs=100, verbose=True,
                    z_dim=2,
                    learning_rate_autoencoder=0.01,
                    learning_rate_discriminator=0.01,
                    learning_rate_generator=0.01,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

    return

    n_neurons_of_hidden_layer_x_autoencoder = \
        create_random_network_architectures(max_layers=3, init_n_neurons=3000,
                                            n_neurons_decay_factors=[1, 1.5, 2, 3])

    n_neurons_of_hidden_layer_x_discriminator = \
        create_random_network_architectures(max_layers=3, init_n_neurons=3000,
                                            n_neurons_decay_factors=[1, 1.5, 2, 3])

    bias_init_value_of_hidden_layer_x_autoencoder = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_autoencoder]
    bias_init_value_of_hidden_layer_x_discriminator = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_discriminator]

    do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=[100], verbose=False,
                  z_dim=[2],
                  learning_rate_autoencoder=[0.001],
                  learning_rate_discriminator=[0.01],
                  learning_rate_generator=[0.01],
                  AdamOptimizer_beta1_autoencoder=[0.5],
                  AdamOptimizer_beta1_discriminator=[0.5],
                  AdamOptimizer_beta1_generator=[0.5],
                  n_neurons_of_hidden_layer_x_autoencoder=n_neurons_of_hidden_layer_x_autoencoder,
                  n_neurons_of_hidden_layer_x_discriminator=n_neurons_of_hidden_layer_x_discriminator,
                  bias_init_value_of_hidden_layer_x_autoencoder=[0.0],
                  bias_init_value_of_hidden_layer_x_discriminator=[0.0])


    return

    n_neurons_of_hidden_layer_x_autoencoder = [[3000, 2000, 1000, 500, 250, 125], [3000, 3000]]
    n_neurons_of_hidden_layer_x_discriminator = [[3000, 2000, 1000, 500, 250, 125], [3000, 3000]]
    bias_init_value_of_hidden_layer_x_autoencoder = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_autoencoder]
    bias_init_value_of_hidden_layer_x_discriminator = [[0.0]*(len(i)+1) for i in
                                                       n_neurons_of_hidden_layer_x_discriminator]

    print(n_neurons_of_hidden_layer_x_autoencoder)
    print(n_neurons_of_hidden_layer_x_discriminator)
    print(bias_init_value_of_hidden_layer_x_autoencoder)
    print(bias_init_value_of_hidden_layer_x_discriminator)

    return

    do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=[1,2], verbose=True,
                  n_neurons_of_hidden_layer_x_autoencoder=[[3000, 2000, 1000, 500, 250, 125]],
                  n_neurons_of_hidden_layer_x_discriminator=[[3000, 2000, 1000, 500, 250, 125]],
                  bias_init_value_of_hidden_layer_x_autoencoder=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                  bias_init_value_of_hidden_layer_x_discriminator=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    return

    """
    test random search
    """

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="MNIST", n_epochs=100, z_dim=10,
                    verbose=True,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return

    # do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=10, z_dim=2,
    #                 verbose=True, learning_rate_autoencoder=0.01, learning_rate_discriminator=0.01,
    #                 learning_rate_generator=0.01)
    #
    # do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=10, z_dim=15,
    #                 verbose=True)

    do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=2000, z_dim=2,
                    verbose=True, batch_size=100,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[1000, 1000],
                    n_neurons_of_hidden_layer_x_discriminator=[1000, 1000],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0])

    # do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="MNIST", n_epochs=100, z_dim=15,
    #                 verbose=True)

    # do_randomsearch(2, selected_autoencoder="SemiSupervised", selected_dataset="MNIST", n_epochs=100, z_dim=15,
    #                 verbose=True, n_neurons_of_hidden_layer_x_autoencoder=[1536, 768, 384],
    #                 n_neurons_of_hidden_layer_x_discriminator=[1536, 768, 384],
    #                 bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0],
    #                 bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0])

    # do_randomsearch(1, selected_autoencoder="Supervised", selected_dataset="MNIST", z_dim=2)

    # do_randomsearch(10, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="Supervised", selected_dataset="SVHN",
    #                 z_dim=15, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=10, verbose=False)

    # do_randomsearch(10, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="Supervised", selected_dataset="SVHN",
    #                 z_dim=100, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=100, verbose=False)


    # do_randomsearch(100, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="SemiSupervised", selected_dataset="SVHN",
    #                 z_dim=15, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=100, verbose=False)


    return

    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # print(aae_parameter_class.draw_from_distribution(distribution_name="normal", loc=1.0, scale=2.0, n_samples=1))
    # print(aae_parameter_class.draw_from_distribution(distribution_name="uniform", low=50, high=500))
    # print(np.random.normal(loc=1.0, scale=2.0))
    # aae_parameter_class.get_randomized_parameters(batch_size={"distribution_name": "uniform", "low":50, "high":500})

    do_randomsearch(7)

    do_randomsearch(8, batch_size={"distribution_name": "uniform", "low": 50, "high": 500, "return_type": "int"})

    do_randomsearch(9, "AdamOptimizer_beta1_discriminator",
                    batch_size={"distribution_name": "normal", "loc": 150.0, "scale": 50.0, "return_type": "int",
                                "is_greater_than_zero": True},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"},
                    RMSPropOptimizer_centered_autoencoder=False)

    do_randomsearch(10, batch_size={"distribution_name": "normal", "loc": 150.0, "scale": 50.0, "return_type": "int",
                                    "is_smaller_than_zero": True},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"})

    do_randomsearch(11, batch_size={"distribution_name": "uniform", "low": 50, "high": 500, "return_type": "int"},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"},
                    RMSPropOptimizer_centered_autoencoder=False, optimizer_autoencoder="ProximalAdagradOptimizer",
                    n_neurons_of_hidden_layer_x_autoencoder=[2587, 237, 29357])

    # do_randomsearch(100)
    # do_randomsearch(2, "batch_size", learning_rate_autoencoder=[random.uniform(0.2, 0.001)*9])
    # do_randomsearch(10, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
    # do_randomsearch(5, "batch_size", "learning_rate_autoencoder")
    # do_randomsearch(10, batch_size=random.uniform(0.2, 0.001))
    # do_randomsearch(5, learning_rate_autoencoder=random.uniform(0.2, 0.001),
    #                 learning_rate_discriminator=random.uniform(0.2, 0.001))

    return

    """
    test grid search
    """
    do_gridsearch(n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                  n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])

    do_gridsearch(n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", "learning_rate_autoencoder")

    do_gridsearch()

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5],
                  MomentumOptimizer_momentum_autoencoder=[1.0])

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5, 0.1, 0.01, 0.001],
                  MomentumOptimizer_momentum_autoencoder=[1.0])


if __name__ == '__main__':
    testing()
