import os
import threading
import uuid

import connexion
import sys

from swagger_server.utils.Storage import Storage
from util.AdversarialAutoencoderParameters import get_result_path_for_selected_autoencoder, get_default_parameters
from util.TuningFunctions import do_randomsearch, do_gridsearch, set_tuning_status


def stop_tuning():
    """
    stops the tuning
    :return:
    """

    # stop the tuning
    set_tuning_status("stop")

    # get the adv. autoencoder
    if Storage.get_aae():
        aae = Storage.get_aae()
    else:
        return "No autoencoder found", 404

    # stop the training of the current autoencoder
    aae.set_train_status("stop")

    return "Tuning stopped", 200


def run_randomsearch(aae_parameters, selected_autoencoder, n_randomized_parameter_combinations):
    """
    runs a random search using n_randomized_parameter_combinations different parameter combinations. Provided parameter
    values, e.g. batch_size=100, will be used throughout all of the parameter combinations, whereas "missing" parameters
    will be used with randomized values.
    :param aae_parameters: parameter values shared by all runs
    :param selected_autoencoder: what autoencoder to use
    :param n_randomized_parameter_combinations: how many combinations should be evaluated
    :return:
    """

    if connexion.request.is_json:

        # get the parameters for the adv autoencoder
        aae_parameters = connexion.request.get_json()

        # check if we have a dataset selected
        if not Storage.get_selected_dataset():
            return "Error: data set not found", 404

        # get the selected dataset ["MNIST", "SVHN", "cifar10", "custom"]
        selected_dataset = Storage.get_selected_dataset()
        aae_parameters["selected_dataset"] = selected_dataset

        # set the results_path based on the selected autoencoder and the selected autoencoder
        aae_parameters["results_path"] = get_result_path_for_selected_autoencoder(selected_autoencoder)
        aae_parameters["selected_autoencoder"] = selected_autoencoder

        # check which params are missing; those will then be used for randomizing the parameters
        default_params = get_default_parameters(selected_autoencoder, selected_dataset)
        params_selected_for_random_search = [key for key in default_params if key not in aae_parameters]
        print("params selected as args for random search: \n", params_selected_for_random_search)

        # we need a tuple as input for args
        args = tuple([n_randomized_parameter_combinations] + params_selected_for_random_search)

        try:
            tuning_thread = threading.Thread(target=do_randomsearch, args=args,
                                             kwargs=aae_parameters)
            tuning_thread.start()

        except KeyError:
            return 'Error: Parameter %s not found' % sys.exc_info()[1], 404

        # store the parameters and the selected autoencoder in the storage class
        Storage.set_aae_parameters(aae_parameters)
        Storage.set_selected_autoencoder(selected_autoencoder)

        return "Success: random search has started", 200

    return 'Error: parameters not in .json format', 415


def run_gridsearch(aae_parameters, selected_autoencoder):
    """
    runs a random search using n_randomized_parameter_combinations different parameter combinations. Provided parameter
    values, e.g. batch_size=100, will be used throughout all of the parameter combinations, whereas "missing" parameters
    will be used with randomized values.
    :param aae_parameters: parameter values shared by all runs
    :param selected_autoencoder: what autoencoder to use
    :return:
    """

    if connexion.request.is_json:

        # get the parameters for the adv autoencoder
        aae_parameters = connexion.request.get_json()

        # check if we have a dataset selected
        if not Storage.get_selected_dataset():
            return "Error: data set not found", 404

        # get the selected dataset ["MNIST", "SVHN", "cifar10", "custom"]
        selected_dataset = Storage.get_selected_dataset()
        aae_parameters["selected_dataset"] = selected_dataset

        # set the results_path based on the selected autoencoder and the selected autoencoder
        aae_parameters["results_path"] = get_result_path_for_selected_autoencoder(selected_autoencoder)
        aae_parameters["selected_autoencoder"] = selected_autoencoder

        print(aae_parameters)

        try:
            tuning_thread = threading.Thread(target=do_gridsearch, kwargs=aae_parameters)
            tuning_thread.start()

        except KeyError:
            return 'Error: Parameter %s not found' % sys.exc_info()[1], 404

        # store the parameters and the selected autoencoder in the storage class
        Storage.set_aae_parameters(aae_parameters)
        Storage.set_selected_autoencoder(selected_autoencoder)

        return "Success: grid search has started", 200

    return 'Error: parameters not in .json format', 415


def get_tuning_results():
    """
    returns the tuning results as list of dictionaries ordered by their total loss:
    [{"parameter_combination": {...}, "performance": {"loss_x": x, "loss_y": y, ..}, "folder_name": "some path"}, ...]
    :return:
    """

    # check if we have
    if Storage.get_tuning_results():

        # save the tuning results for the swagger server
        tuning_results = Storage.get_tuning_results()

        return tuning_results, 200

    else:
        return "No tuning results found", 404


