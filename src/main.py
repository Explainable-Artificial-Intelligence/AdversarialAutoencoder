import os

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
    get_result_path_for_selected_autoencoder, get_default_parameters_svhn
from util.TuningFunctions import do_randomsearch, do_gridsearch, init_aae_with_params_file

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

    if True:
        param_search_incorporating_label_information()

        return




    """
    test yeast mass spec data:
    """
    if False:

        params = get_default_parameters_mnist()

        params["input_dim_x"] = 1
        params["input_dim_y"] = 152
        params["n_classes"] = 2

        params["n_epochs"] = 100

        params["selected_dataset"] = "yeast_mass_spec"

        params["verbose"] = True
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")

        # aae = LearningPriorsAdversarialAutoencoderUnsupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSupervised(params)
        # aae = LearningPriorsAdversarialAutoencoderSameTopology(params)
        # aae = LearningPriorsAdversarialAutoencoder(params)
        aae = UnsupervisedAdversarialAutoencoder(params)

        aae.train(True)

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

        params["z_dim"] = 2

        params["verbose"] = True
        params["selected_autoencoder"] = "Unsupervised"
        params["results_path"] = get_result_path_for_selected_autoencoder("Unsupervised")
        params["summary_image_frequency"] = 1
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

        params["verbose"] = True
        params["selected_autoencoder"] = "IncorporatingLabelInformation"
        params["results_path"] = get_result_path_for_selected_autoencoder("IncorporatingLabelInformation")
        params["n_labeled"] = 20000
        params["summary_image_frequency"] = 5
        params["n_epochs"] = 101
        params["batch_normalization_encoder"] = [None, None, None, None, None]
        params["batch_normalization_decoder"] = [None, None, None, None, None]
        params["batch_normalization_discriminator"] = [None, None, None, None, None]

        params["save_final_model"] = True

        params["n_neurons_of_hidden_layer_x_autoencoder"] = [1000, 1000]
        params["n_neurons_of_hidden_layer_x_discriminator"] = [1000, 1000]

        params["activation_function_encoder"] = ['relu', 'relu', 'linear']
        params["activation_function_decoder"] = ['relu', 'relu', 'sigmoid']
        params["activation_function_discriminator"] = ['relu', 'relu', 'linear']

        params["loss_function_generator"] = "sigmoid_cross_entropy"
        params["loss_function_discriminator"] = "sigmoid_cross_entropy"

        params["dropout_encoder"] = [0.0, 0.0, 0.0]
        params["dropout_decoder"] = [0.0, 0.0, 0.0]

        params["decaying_learning_rate_name_autoencoder"] = "static"
        params["decaying_learning_rate_name_generator"] = "static"
        params["decaying_learning_rate_name_discriminator"] = "static"

        params["decaying_learning_rate_params_autoencoder"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_generator"] = {"learning_rate": 0.0001}
        params["decaying_learning_rate_params_discriminator"] = {"learning_rate": 0.0001}

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
        create_random_network_topologies(max_layers=3, init_n_neurons=3000,
                                         n_neurons_decay_factors=[1, 1.5, 2, 3])

    n_neurons_of_hidden_layer_x_discriminator = \
        create_random_network_topologies(max_layers=3, init_n_neurons=3000,
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
