from tensorflow.contrib.learn.python.learn.datasets import base

from src.util.DataLoading import get_input_data
from swagger_server.utils.Storage import Storage


def load_data_set(dataset_name):
    """
    loads a dataset into the storage class
    :param dataset_name: one of ["MNIST", "SVHN", "cifar10", "custom"]
    :return:
    """

    if dataset_name not in ["MNIST", "SVHN", "cifar10", "custom"]:
        return "dataset name not found", 404

    # TODO: filepath for custom dataset
    dataset = get_input_data(dataset_name, filepath="../data")

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
    batch_dict = {"images:": batch_images.astype("float64").tolist(), "labels:": batch_labels.tolist()}

    return batch_dict, 200


