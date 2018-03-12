import connexion
import numpy as np
import time

from swagger_server.utils.Storage import Storage


def generate_image_from_single_point(single_point):
    """
    generates the AE output from a given point of the sample distribution

    :param
    :type

    :rtype:
    """

    if not Storage.get_aae_parameters():
        return "Error: autoencoder not found", 404
    z_dim = Storage.get_aae_parameters()["z_dim"]

    if Storage.get_selected_autoencoder() != "Unsupervised":
        return "Error: This function is supposed to work for unsupervised autoencoders only!", 412

    if len(single_point) != z_dim:
        return "Error: Invalid dimension! Dimension should be %s." % z_dim, 400

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: autoencoder not found", 404

    # create the operation for the aae and add it
    operation = {"generate_image_from_single_point": single_point}
    aae.add_to_requested_operations_by_swagger(operation)

    # training has already stopped ..
    if aae.get_train_status() == "stop":
        # .. so we restart the aae
        aae.train(False)

    # wait for the aae to process the operation
    while aae.get_requested_operations_by_swagger_results() is None:
        # wait for 200 ms, then check again
        time.sleep(0.2)

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = result.astype("float64").tolist()

    return result, 200


def generate_image_from_single_point_and_single_label(single_point, class_label):
    """
    generates the AE output from a given point of the sample distribution

    :param
    :type

    :rtype:
    """

    if not Storage.get_aae_parameters():
        return "Error: autoencoder not found", 404
    z_dim = Storage.get_aae_parameters()["z_dim"]

    if Storage.get_selected_autoencoder() == "Unsupervised":
        return "Error: This function is supposed to work for (semi-)supervised autoencoders only!", 412

    if len(single_point) != z_dim:
        return "Error: Invalid dimension! Dimension should be %s." % z_dim, 400

    if class_label > Storage.get_aae_parameters()["n_classes"] + 1:
        return "Error: Invalid class label! Should can be at most %i" \
               % Storage.get_aae_parameters()["n_classes"] + 1, 400

    # convert single label to one-hot vector
    one_hot_label = [0] * Storage.get_aae_parameters()["n_classes"]
    one_hot_label[class_label - 1] = 1

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: autoencoder not found", 404

    operation = {"generate_image_from_single_point_and_single_label": (single_point, one_hot_label)}
    aae.add_to_requested_operations_by_swagger(operation)

    # training is still in progress
    if aae.get_train_status() == "stop":
        #
        aae.train(False)

    # wait for the response from the aae
    while aae.get_requested_operations_by_swagger_results() is None:
        # wait for 200 ms, then check again
        time.sleep(0.2)

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = result.astype("float64").tolist()

    return result, 200


def generate_image_grid():
    """
    generates the AE output from a given point of the sample distribution

    :param
    :type

    :rtype:
    """

    # check if we have an autoencoder
    if not Storage.get_aae():
        return "Error: autoencoder not found", 404

    # get the autoencoder
    aae = Storage.get_aae()

    # we don't need a parameter for the image grid
    operation = {"generate_image_grid": ""}
    aae.add_to_requested_operations_by_swagger(operation)

    # training is still in progress
    if aae.get_train_status() == "stop":
        #
        aae.train(False)

    # wait for the response from the aae
    while aae.get_requested_operations_by_swagger_results() is None:
        # wait for 200 ms, then check again
        time.sleep(0.2)

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = [a.astype("float64").tolist() for a in result]

    return result, 200




