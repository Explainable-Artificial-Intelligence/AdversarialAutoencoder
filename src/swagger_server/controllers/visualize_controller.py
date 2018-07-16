import time


from swagger_server.utils.Storage import Storage
from swagger_server.utils.SwaggerUtils import convert_image_array_to_byte_string


def generate_image_from_single_point(single_point):
    """
    generates the aae output from a given point on the latent space
    :param single_point: point on the latent space
    :return: 2 or 3D (depending on colorscale) list holding the image pixels
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

    total_waiting_time = 0

    # wait for the response from the aae (at maximum 30s)
    while aae.get_requested_operations_by_swagger_results() is None and total_waiting_time <= 30:
        # wait for 200 ms, then check again
        time.sleep(0.2)
        total_waiting_time += 0.2

    # response took too long..
    if total_waiting_time > 30:
        return "Request timed out", 408

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = result.astype("float64").tolist()

    return result, 200


def generate_image_from_single_point_as_byte_string(single_point):
    """
    generates the aae output from a given point on the latent space and returns it as byte string
    :param single_point: point on the latent space
    :return: byte string of the image
    """

    img, response_code = generate_image_from_single_point(single_point)
    byte_string = convert_image_array_to_byte_string(img, channels=Storage.get_n_channels())
    return byte_string, response_code


def generate_image_from_single_point_and_single_label(single_point, class_label):
    """
    generates the aae output from a given point on the latent space and a given class label
    :param single_point: point on the latent space
    :param class_label: integer
    :return: 2 or 3D (depending on colorscale) list holding the image pixels
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

    total_waiting_time = 0

    # wait for the response from the aae (at maximum 30s)
    while aae.get_requested_operations_by_swagger_results() is None and total_waiting_time <= 30:
        # wait for 200 ms, then check again
        time.sleep(0.2)
        total_waiting_time += 0.2

    # response took too long..
    if total_waiting_time > 30:
        return "Request timed out", 408

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = result.astype("float64").tolist()

    return result, 200


def generate_image_from_single_point_and_single_label_as_byte_string(single_point, class_label):
    """
    generates the aae output from a given point on the latent space and a given class label and returns it as byte
    string
    :param single_point: point on the latent space
    :param class_label: integer
    :return: byte string of the image
    """
    img, response_code = generate_image_from_single_point_and_single_label(single_point, class_label)
    byte_string = convert_image_array_to_byte_string(img, channels=Storage.get_n_channels())
    return byte_string, response_code


def generate_image_grid():
    """
    generates the image grid by sampling the latent space and returning it
    :return:
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

    total_waiting_time = 0

    # wait for the response from the aae (at maximum 30s)
    while aae.get_requested_operations_by_swagger_results() is None and total_waiting_time <= 30:
        # wait for 200 ms, then check again
        time.sleep(0.2)
        total_waiting_time += 0.2

    # response took too long..
    if total_waiting_time > 30:
        return "Request timed out", 408

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = [a.astype("float64").tolist() for a in result]

    return result, 200


def generate_image_grid_as_byte_string():
    """
    generates the image grid by sampling the latent space and returning it as array of byte strings
    :return:
    """

    imgs, response_code = generate_image_grid()
    list_of_byte_strings = [convert_image_array_to_byte_string(img, channels=Storage.get_n_channels()) for img in imgs]
    return list_of_byte_strings, response_code


def classify_single_image(single_image):
    """
    classifies a single image and returns the predicted class label as integer label
    :param single_image: numpy array of the image to classify
    :return: integer label of the predicted class
    """

    if not Storage.get_aae_parameters():
        return "Error: autoencoder not found", 404
    input_dim_x = Storage.get_aae_parameters()["input_dim_x"]
    input_dim_y = Storage.get_aae_parameters()["input_dim_y"]
    color_scale = Storage.get_aae_parameters()["color_scale"]
    input_dim = 0

    if color_scale == "gray_scale":
        input_dim = input_dim_x*input_dim_y
    elif color_scale == "rgb_scale":
        input_dim = input_dim_x*input_dim_y*3

    if not Storage.get_selected_autoencoder() == "SemiSupervised":
        return "Error: This function is supposed to work for semi-supervised autoencoders only!", 412

    if len(single_image) != input_dim:
        return "Error: Invalid dimension! Dimension should be %s." % input_dim, 400

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: autoencoder not found", 404

    operation = {"classify_single_image": single_image}
    aae.add_to_requested_operations_by_swagger(operation)

    # training has stopped
    if aae.get_train_status() == "stop":
        # restart aae and predict the label
        aae.train(False)

    total_waiting_time = 0

    # wait for the response from the aae (at maximum 30s)
    while aae.get_requested_operations_by_swagger_results() is None and total_waiting_time <= 30:
        # wait for 200 ms, then check again
        time.sleep(0.2)
        total_waiting_time += 0.2

    # response took too long..
    if total_waiting_time > 30:
        return "Request timed out", 408

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np ints are not json serializable
    result = int(result)

    return result, 200


def get_biases_or_weights_for_layer(bias_or_weights, layer_name):

    # TODO: layer name as enum

    if bias_or_weights not in ["bias", "weights"]:
        return "invalid input", 400

    if not Storage.get_aae_parameters():
        return "Error: autoencoder not found", 404

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: autoencoder not found", 404

    # check if the layer_name is valid
    subnetwork = layer_name.split("_")[0]      # encoder, decoder, etc
    all_layer_names = aae.get_all_layer_names()
    try:
        all_layer_names[subnetwork]
    except KeyError:
        return "Error: layer_name is invalid!", 400
    if layer_name not in all_layer_names[subnetwork]:
        return "Error: layer_name is invalid!", 400

    # request the operation
    operation = {"get_biases_or_weights_for_layer": (bias_or_weights, layer_name)}
    aae.add_to_requested_operations_by_swagger(operation)

    # training has stopped
    if aae.get_train_status() == "stop":
        # restart aae and get the weights/biases
        aae.train(False)

    total_waiting_time = 0

    # wait for the response from the aae (at maximum 30s)
    while aae.get_requested_operations_by_swagger_results() is None and total_waiting_time <= 30:
        # wait for 200 ms, then check again
        time.sleep(0.2)
        total_waiting_time += 0.2

    # response took too long..
    if total_waiting_time > 30:
        return "Request timed out! Maybe you need to start training first", 408

    # aae has responded
    result = aae.get_requested_operations_by_swagger_results()

    # reset the variable holding the results
    aae.set_requested_operations_by_swagger_results(None)

    # we need to convert it, since np arrays are not json serializable
    result = [a.astype("float64").tolist() for a in result]

    return result

