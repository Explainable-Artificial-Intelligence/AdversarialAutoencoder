import json

import datetime

import util.AdversarialAutoencoderParameters as aae_params
import numpy as np

from autoencoders.SemiSupervisedAdversarialAutoencoder import SemiSupervisedAdversarialAutoencoder
from autoencoders.SupervisedAdversarialAutoencoder import SupervisedAdversarialAutoencoder
from autoencoders.UnsupervisedAdversarialAutoencoder import UnsupervisedAdversarialAutoencoder
from swagger_server.utils.Storage import Storage


tuning_status = "start"


def get_tuning_status():
    return tuning_status


def set_tuning_status(tuning_status_to_set):
    global tuning_status
    tuning_status = tuning_status_to_set


def init_aae_with_params_file(params_filename, used_aae):
    """
    returns a adversarial autoencoder initiated with the provided parameter file
    :param params_filename: path to the params.txt file
    :param used_aae: ["Unsupervised", "Supervised", "SemiSupervised"]
    :return:
    """

    # get the used parameters from the params.txt file
    used_params = json.load(open(params_filename))

    # compability with old params file
    if used_params.get("write_tensorboard") is None:
        used_params["write_tensorboard"] = False
    if used_params.get("summary_image_frequency") is None:
        used_params["summary_image_frequency"] = 10

    # create the AAE and train it with the used parameters
    adv_autoencoder = None
    if used_aae == "Unsupervised":
        adv_autoencoder = UnsupervisedAdversarialAutoencoder(used_params)
    elif used_aae == "Supervised":
        adv_autoencoder = SupervisedAdversarialAutoencoder(used_params)
    elif used_aae == "SemiSupervised":
        adv_autoencoder = SemiSupervisedAdversarialAutoencoder(used_params)

    return adv_autoencoder


def do_gridsearch(*args, selected_autoencoder="Unsupervised", selected_dataset="MNIST", **kwargs):
    """
    Performs a grid search using all possible combinations of the parameters provided. In case there are no parameters
    provided it uses all the possible parameter combinations from the hard coded parameters.
    Example calls:
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.1, 0.01, 0.001],
                    MomentumOptimizer_momentum_autoencoder=[1.0, 0.9, 0.8])
        - do_gridsearch(n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                        n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])
        - do_gridsearch(n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", "learning_rate_autoencoder")
        - do_gridsearch()
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5],
                        MomentumOptimizer_momentum_autoencoder=[1.0])
    :param args: strings of the variable defined in the Parameters class to do the grid search for. In this case
    it uses the possible parameter values in the Parameters class: "variable_name"
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param kwargs: arbitrary number of: variable_name=[variable_value1, variable_value2, variable_value3]
    :return: the best parameter combination as a dictionary
    """

    print("Doing grid search..")

    log_result_path = "../results/Logs/GridSearch"
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", ":").replace(":", "_")
    log_file_name = log_result_path + "/{0}_{1}_log.txt".format(date, selected_dataset)

    print("Log will be saved at location " + log_file_name)

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_params.get_gridsearch_parameters(*args, selected_autoencoder=selected_autoencoder,
                                             selected_dataset=selected_dataset, **kwargs)

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(gridsearch_parameter_combinations), "combinations:")

    for a in gridsearch_parameter_combinations:
        print(a)
    print()

    # iterate over each parameter combination
    for gridsearch_parameter_combination in gridsearch_parameter_combinations:

        # for controlling the tuning via swagger
        if not tuning_status == "stop":

            print("Training .. ", gridsearch_parameter_combination)

            # create the AAE and train it with the current parameters
            if selected_autoencoder == "Unsupervised":
                adv_autoencoder = UnsupervisedAdversarialAutoencoder(gridsearch_parameter_combination)
            elif selected_autoencoder == "Supervised":
                adv_autoencoder = SupervisedAdversarialAutoencoder(gridsearch_parameter_combination)
            elif selected_autoencoder == "SemiSupervised":
                adv_autoencoder = SemiSupervisedAdversarialAutoencoder(gridsearch_parameter_combination)

            # we want to include the results from our previous runs on the minibatch summary images
            adv_autoencoder.set_include_tuning_performance(True)

            # set the autoencoder for the swagger server
            Storage.set_aae(adv_autoencoder)

            # start the training
            adv_autoencoder.train(True)
            # adv_autoencoder.train(False)

            # get the performance
            performance = adv_autoencoder.get_final_performance()
            print(performance)

            # convert performance to float64 (for swagger server)
            for key, value in performance.items():
                performance[key] = np.float64(value)

            folder_name = adv_autoencoder.get_result_folder_name()

            # store the param_comb and the performance in the list
            current_performance = {"parameter_combination": gridsearch_parameter_combination,
                                   "performance": performance, "folder_name": folder_name}
            performance_for_parameter_combination.append(current_performance)

            # store the performance over time of the current autoencoder
            Storage.get_tuning_results_performance_over_time()[folder_name] = \
                adv_autoencoder.get_performance_over_time()

            # store the learning rates over time of the current autoencoder
            Storage.get_tuning_results_learning_rates_over_time()[folder_name] = adv_autoencoder.get_learning_rates()

            # reset the tensorflow graph
            adv_autoencoder.reset_graph()

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x["performance"]["summed_loss_final"])

    # save the tuning results for the swagger server
    Storage.set_tuning_results(sorted_list)

    print("#" * 20)

    # create a new log file
    with open(log_file_name, 'w') as log:
        log.write("")

    for comb in sorted_list:
        print("performance:", comb["performance"])
        print("folder name:", comb["folder_name"])
        print()
        with open(log_file_name, 'a') as log:
            log.write("performance: {}\n".format(comb["performance"]))
            log.write("folder name: {}\n".format(comb["folder_name"]))

    print(sorted_list)
    print("best param combination:", sorted_list[0]["parameter_combination"])
    print("best performance:", sorted_list[0]["performance"])
    print("folder name:", sorted_list[0]["folder_name"])

    with open(log_file_name, 'a') as log:
        log.write("best param combination: {}\n".format(sorted_list[0]["parameter_combination"]))
        log.write("best performance: {}\n".format(sorted_list[0]["performance"]))
        log.write("folder name: {}\n".format(sorted_list[0]["folder_name"]))

    return sorted_list[0]["parameter_combination"]


def do_randomsearch(n_parameter_combinations=5, *args, selected_autoencoder="Unsupervised", selected_dataset="MNIST",
                    **kwargs):
    """
    Performs a random search using n_parameter_combinations different parameter combinations. The parameter combination
    is obtained by randomly assigning values for the parameters provided (args and kwargs).
    Example calls:
        - do_randomsearch()
        - do_randomsearch(2, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
        - do_randomsearch(10, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
        - do_randomsearch(5, "batch_size", "learning_rate_autoencoder")
        - do_randomsearch(5, learning_rate_autoencoder=random.uniform(0.2, 0.001),
                          learning_rate_discriminator=random.uniform(0.2, 0.001))
    :param n_parameter_combinations: number of parameter combinations to try
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param args: strings of the variable defined in the Parameters class to randomize
    :param kwargs: manually assigned values for the specified variable
    :return: the best parameter combination as a dictionary
    """

    print("Doing random search..")

    log_result_path = "../results/Logs/RandomSearch"
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", ":").replace(":", "_")
    log_file_name = log_result_path + "/{0}_{1}_log.txt".format(date, selected_dataset)

    print("Log will be saved at location " + log_file_name)

    # get some random parameter combinations
    random_param_combinations = \
        [aae_params.get_randomized_parameters(*args, selected_autoencoder=selected_autoencoder,
                                              selected_dataset=selected_dataset, **kwargs)
         for i in range(n_parameter_combinations)]

    # TODO: think about this, whether it should be included all the time
    # add the default parameter combination to the list based on the selected dataset
    # random_param_combinations.append(aae_parameter_class.get_default_parameters(selected_autoencoder, selected_dataset))

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(random_param_combinations), "combinations:")
    for a in random_param_combinations:
        print(a)
    print()

    # iterate over each parameter combination
    for random_param_combination in random_param_combinations:

        # for controlling the tuning via swagger
        if not tuning_status == "stop":

            print(random_param_combination)

            # create the AAE and train it with the current parameters
            if selected_autoencoder == "Unsupervised":
                adv_autoencoder = UnsupervisedAdversarialAutoencoder(random_param_combination)
            elif selected_autoencoder == "Supervised":
                adv_autoencoder = SupervisedAdversarialAutoencoder(random_param_combination)
            elif selected_autoencoder == "SemiSupervised":
                adv_autoencoder = SemiSupervisedAdversarialAutoencoder(random_param_combination)

            # we want to include the results from our previous runs on the minibatch summary images
            adv_autoencoder.set_include_tuning_performance(True)
            try:

                # set the autoencoder for the swagger server
                Storage.set_aae(adv_autoencoder)

                # start the training
                adv_autoencoder.train(True)

                # get the performance
                performance = adv_autoencoder.get_final_performance()
            except:
                print("whoops")
                performance = {"autoencoder_loss_final": float('inf'),
                                  "discriminator_loss_final": float('inf'),
                                  "generator_loss_final": float('inf'),
                                  "summed_loss_final": float('inf')}

            print(performance)

            # convert performance to float64 (for swagger server)
            for key, value in performance.items():
                performance[key] = np.float64(value)

            folder_name = adv_autoencoder.get_result_folder_name()

            # store the parameter combination and the performance in the list
            current_performance = {"parameter_combination": random_param_combination,
                                   "performance": performance, "folder_name": folder_name}
            performance_for_parameter_combination.append(current_performance)

            # store the performance over time of the current autoencoder
            Storage.get_tuning_results_performance_over_time()[folder_name] \
                = adv_autoencoder.get_performance_over_time()

            # store the learning rates over time of the current autoencoder
            Storage.get_tuning_results_learning_rates_over_time()[folder_name] \
                = adv_autoencoder.get_learning_rates()

            # reset the tensorflow graph
            adv_autoencoder.reset_graph()

    # sort combinations by their performance
    # TODO: change back to summed loss
    # sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x["performance"]["summed_loss_final"])
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x["performance"]["autoencoder_loss_final"])

    # store the tuning results for the swagger server
    Storage.set_tuning_results(performance_for_parameter_combination)

    print("#" * 20)

    print(Storage.get_tuning_results_performance_over_time())

    # create a new log file
    with open(log_file_name, 'w') as log:
        log.write("")

    for comb in sorted_list:
        print("performance:", comb["performance"])
        print("folder name:", comb["folder_name"])
        print()
        with open(log_file_name, 'a') as log:
            log.write("performance: {}\n".format(comb["performance"]))
            log.write("folder name: {}\n".format(comb["folder_name"]))

    print(sorted_list)
    print("best param combination:", sorted_list[0]["parameter_combination"])
    print("best performance:", sorted_list[0]["performance"])
    print("folder name:", sorted_list[0]["folder_name"])

    with open(log_file_name, 'a') as log:
        log.write("best param combination: {}\n".format(sorted_list[0]["parameter_combination"]))
        log.write("best performance: {}\n".format(sorted_list[0]["performance"]))
        log.write("folder name: {}\n".format(sorted_list[0]["folder_name"]))

    return sorted_list[0]["parameter_combination"]


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

    random_network_topology = [init_n_neurons]*n_layers
    for decaying_layer in range(n_decaying_layers, 0, -1):
        random_network_topology[n_layers-decaying_layer] = \
             int(random_network_topology[n_layers-decaying_layer-1] / n_neurons_decay_factor)

    return random_network_topology


def create_random_network_topologies(max_layers, init_n_neurons, n_neurons_decay_factors):

    random_network_topologies = []

    for n_layers in range(1, max_layers+1):
        # maximum number of layers with a reduced number of nerons compared to the preceding layers
        for n_decaying_layers in range(n_layers):

            # we only have to iterate over the n_neurons_decay_factors if we have at least one decaying layer
            if n_decaying_layers > 0:
                for n_neurons_decay_factor in n_neurons_decay_factors:
                    random_network_topologies.append(
                            create_network_topology(n_layers, init_n_neurons, n_neurons_decay_factor, n_decaying_layers))
            # otherwise we can pick an arbitrary value for the n_neurons_decay_factor
            else:
                random_network_topologies.append(
                    create_network_topology(n_layers, init_n_neurons, 1, n_decaying_layers))

    return random_network_topologies
