from src.util.DataLoading import get_input_data
from swagger_server.utils.Storage import Storage
from swagger_server.utils.SwaggerUtils import convert_image_array_to_byte_string


def load_data_set(dataset_name, mass_spec_data_properties=None):
    """
    loads a dataset into the storage class
    :param dataset_name: one of ["MNIST", "SVHN", "cifar10", "custom"]
    :param mass_spec_data_properties: dictionary holding the properties for the mass spec data
    :return:
    """

    if dataset_name not in ["MNIST", "SVHN", "cifar10", "mass_spec", "custom"]:
        return "dataset name not found", 404

    if dataset_name == "mass_spec" and mass_spec_data_properties is None:
        return "Bad request! mass_spec_data_properties needs to be provided when using mass spec data!", 404

    dataset = get_input_data(dataset_name, filepath="../data", mass_spec_data_properties=mass_spec_data_properties)

    # store the data in the storage class
    Storage.set_input_data(dataset)
    Storage.set_selected_dataset(dataset_name)

    return "data successfully loaded", 200


def get_single_image(image_id, data_subset_name="train"):
    """
    returns a single image with the respective id
    :param image_id: id of the image to return
    :param data_subset_name: one of ["train", "test", "validation"]
    :return:
    """

    try:
        data = Storage.get_input_data(data_subset_name)
    except KeyError:
        return "No data found", 404

    try:
        image = data.images[image_id]
    except IndexError:
        return "Index out of bounds", 415

    return list(image.astype("float64")), 200


def get_single_image_as_byte_string(image_id, data_subset_name="train"):
    """
    returns a single image with the respective id as byte string
    :param image_id: id of the image to return
    :param data_subset_name: one of ["train", "test", "validation"]
    :return:
    """

    img, response_code = get_single_image(image_id, data_subset_name)
    byte_string = convert_image_array_to_byte_string(img, channels=Storage.get_n_channels())
    return byte_string, response_code


def get_single_label(label_id, data_subset_name="train"):
    """
    returns a single label with the respective id
    :param label_id: id of the label to return
    :param data_subset_name: one of ["train", "test", "validation"]
    :return:
    """

    try:
        data = Storage.get_input_data(data_subset_name)
    except KeyError:
        return "No data found", 404

    try:
        label = data.labels[label_id]
    except IndexError:
        return "Index out of bounds", 415

    return list(label), 200


def get_data_batch(batch_size=100, data_subset_name="train"):
    """
    returns the data (images and labels) for the current batch
    :param batch_size: size of the batch
    :param data_subset_name: one of ["train", "test", "validation"]
    :return:
    """

    try:
        data = Storage.get_input_data(data_subset_name)
    except KeyError:
        return "No data found", 404

    # get the images and the labels for the current batch
    batch_images, batch_labels = data.next_batch(batch_size)

    # store current batch_images and labels in the storage class
    Storage.set_current_batch_data({"images": batch_images, "labels": batch_labels})

    # since swagger doesn't allow multiple return values, we have to pack them in a dictionary and return it
    batch_dict = {"images": batch_images.astype("float64").tolist(), "labels": batch_labels.tolist()}

    return batch_dict, 200


def get_data_batch_as_byte_string(batch_size=100, data_subset_name="train"):
    """
    returns the data (images and labels) for the current batch as byte string
    :param batch_size: size of the batch
    :param data_subset_name: one of ["train", "test", "validation"]
    :return:
    """

    batch_dict, response_code = get_data_batch(batch_size, data_subset_name)

    channels = Storage.get_n_channels()
    images = batch_dict["images"]
    images = [convert_image_array_to_byte_string(image, channels) for image in images]

    batch_dict["images"] = images

    return batch_dict, 200

