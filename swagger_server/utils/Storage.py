"""includes a static class which stores all server data"""
import threading


class Storage(object):

    # stores the autoencoder used: ["Unsupervised", "Supervised", "SemiSupervised"]
    selected_autoencoder = None

    # stores the data set used: ["MNIST", "SVHN", "cifar10", "custom"]
    selected_dataset = None

    # stores the data in this dictionary; default keys: train, test and validation
    # train images/labels can then be accessed like this: input_data.train.images/input_data.train.labels
    input_data = {}

    # stores the current adversarial autoencoder
    aae = None

    # stores the parameters used for the adv. autoencoder
    aae_parameters = None

    # stores the data of the current batch
    current_batch_data = {}

    # holds the thread for training the autoencoder
    aae_thread = None

    # holds the tuning results
    tuning_results = None

    # TODO: check which one are needed
    input_batch_indices = {}
    output_data = {}
    output_batch_indices = {}
    train_step = 0

    tuning_ANNs = []
    tuning_thread = threading.Thread()
    tuning_queue = object()

    @classmethod
    def set_tuning_results(cls, tuning_results):
        cls.tuning_results = tuning_results

    @classmethod
    def get_tuning_results(cls):
        return cls.tuning_results

    @classmethod
    def set_selected_autoencoder(cls, selected_autoencoder):
        cls.selected_autoencoder = selected_autoencoder

    @classmethod
    def get_selected_autoencoder(cls):
        return cls.selected_autoencoder

    @classmethod
    def set_current_batch_data(cls, current_batch_data):
        cls.current_batch_data = current_batch_data

    @classmethod
    def get_current_batch_data(cls):
        return cls.current_batch_data

    @classmethod
    def set_selected_dataset(cls, selected_dataset):
        cls.selected_dataset = selected_dataset

    @classmethod
    def get_selected_dataset(cls):
        return cls.selected_dataset

    @classmethod
    def set_input_data(cls, train_data, dataset_name="train"):
        cls.input_data[dataset_name] = train_data

    @classmethod
    def get_input_data(cls, dataset_name="train"):
        return cls.input_data[dataset_name]

    @classmethod
    def set_aae(cls, cae):
        cls.aae = cae

    @classmethod
    def get_aae(cls):
        return cls.aae

    @classmethod
    def set_output_data(cls, datasetname, train_data_output):
        cls.output_data[datasetname] = train_data_output

    @classmethod
    def get_output_data(cls, dataset_name="train"):
        return cls.output_data[dataset_name]

    @classmethod
    def get_output_image(cls, image_id):
        return cls.output_data[image_id]

    @classmethod
    def set_aae_thread(cls, aae_thread):
        cls.aae_thread = aae_thread

    @classmethod
    def get_aae_thread(cls):
        return cls.aae_thread

    @classmethod
    def get_current_batch_index(cls, dataset_name, output):
        if output:
            if not dataset_name in cls.output_batch_indices.keys():
                cls.output_batch_indices[dataset_name] = 0
            return cls.output_batch_indices[dataset_name]
        else:
            if not dataset_name in cls.input_batch_indices.keys():
                cls.input_batch_indices[dataset_name] = 0
            return cls.input_batch_indices[dataset_name]

    @classmethod
    def get_dataset_length(cls, dataset_name, output):
        if output:
            return len(cls.output_data[dataset_name])
        else:
            return len(cls.input_data[dataset_name])

    @classmethod
    def update_batch_index(cls, dataset_name, output, next_batch_index):
        if output:
            cls.output_batch_indices[dataset_name] = next_batch_index
        else:
            cls.input_batch_indices[dataset_name] = next_batch_index

    @classmethod
    def output_images_computed(cls, datasetname):
        return datasetname in cls.output_data.keys()

    @classmethod
    def get_output_image_by_id(cls, datasetname, id):
        dataset = cls.output_data[datasetname]
        return dataset[id]

    @classmethod
    def get_prev_training_step(cls):
        return cls.train_step

    @classmethod
    def update_prev_training_step(cls, set_size):
        cls.train_step += set_size

    @classmethod
    def get_aae_parameters(cls):
        return cls.aae_parameters

    @classmethod
    def set_aae_parameters(cls, aae_parameters):
        cls.aae_parameters = aae_parameters
