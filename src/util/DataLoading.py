import gzip
import collections

import imageio as imageio
import numpy as np
import pandas
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.framework import random_seed
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat
from six.moves import cPickle

import timeit


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.    `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.    Seed arg provides for convenient deterministic test_algorithm.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_images(self, images):
        self._images = images

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def get_class_specific_batch(self, batch_size, class_label, n_classes=10, shuffle=True):
        """
        returns "batch_size" images and labels specific for the class with label=class_label
        :param batch_size: number of images/labels to return
        :param class_label: integer label of class we want to get the samples for
        :param n_classes: number of classes the dataset contains
        :param shuffle: whether to shuffle the data or not
        :return:
        """

        # holds the class specific images
        class_specific_images = []

        # TODO: speed up code

        while len(class_specific_images) < batch_size:
            next_image, next_label_one_hot = self.next_batch(1, shuffle=shuffle)
            # convert label to integer label
            next_label_int = np.argmax(next_label_one_hot, 1)[0]
            # check if it fits
            if next_label_int == class_label:
                class_specific_images.append(next_image)

        # return a numpy array of the images and a numpy array of the one hot labels
        return np.vstack(class_specific_images), np.eye(n_classes)[[class_label]*batch_size]

    def get_color_specific_image_combinations(self, batch_size, label_red, label_green, label_blue,
                                              input_dim_x, input_dim_y, shuffle=True):

        # calculate the number of pixel per channel
        n_colored_pixels_per_channel = input_dim_x * input_dim_y

        # TODO:
        # check the data set used (works only for SVHN or cifar10)

        # get batch_size images for the red channel
        red_channel_images, labels = self.get_class_specific_batch(batch_size, class_label=label_red, shuffle=shuffle)

        # get batch_size images for the red channel
        green_channel_images, _ = self.get_class_specific_batch(batch_size, class_label=label_green, shuffle=shuffle)

        # get batch_size images for the red channel
        blue_channel_images, _ = self.get_class_specific_batch(batch_size, class_label=label_blue, shuffle=shuffle)

        # first n_colored_pixels_per_channel encode red
        red_pixels = red_channel_images[:, :n_colored_pixels_per_channel]
        # next n_colored_pixels_per_channel encode green
        green_pixels = green_channel_images[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2]
        # last n_colored_pixels_per_channel encode blue
        blue_pixels = blue_channel_images[:, n_colored_pixels_per_channel * 2:]

        # concatenate the color arrays into one array
        imgs = np.concatenate([red_pixels, green_pixels, blue_pixels], 1)
        # imgs = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)

        return imgs, labels

"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""


def normalize_np_array(np_array):
    """
    normalizes the given numpy array using z score normalization (subtracting the mean of each element and dividing
    by the std deviation)
    :param np_array:
    :return:
    """
    # get the shape of the data points
    data_point_shape = np_array.shape[1]

    # calculate the mean for each element and create a vector out of it
    mean_per_element = np.mean(np_array, axis=0)
    means_col_vec = mean_per_element.reshape((1, data_point_shape))

    # calculate the std dev for each element and create a vector out of it
    std_dev_per_element = np.std(np_array, axis=0)
    std_dev_col_vec = std_dev_per_element.reshape((1, data_point_shape))

    # apply z score normalization
    return (np_array - means_col_vec) / std_dev_col_vec


def get_input_data(selected_dataset, filepath="../data", color_scale="gray_scale", data_normalized=False):
    """
    returns the input data set based on self.selected_dataset
    :return: object holding the train data, the test data and the validation data
    """

    data = None

    # hand written digits
    if selected_dataset == "MNIST":
        data = read_mnist_data_from_ubyte(filepath, one_hot=True)
    # Street View House Numbers
    elif selected_dataset == "SVHN":
        if color_scale == "gray_scale":
            data = read_svhn_from_mat(filepath, one_hot=True, validation_size=5000, grey_scale=True)
        else:
            data = read_svhn_from_mat(filepath, one_hot=True, validation_size=5000)
    elif selected_dataset == "cifar10":
        if color_scale == "gray_scale":
            data = read_cifar10(filepath, one_hot=True, validation_size=5000, grey_scale=True)
        else:
            data = read_cifar10(filepath, one_hot=True, validation_size=5000)
    elif selected_dataset == "custom":
        # TODO:
        print("not yet implemented")
        raise NotImplementedError

    if data_normalized:

        # get the image data
        train_images= data.train.images
        test_images = data.test.images
        validation_images = data.validation.images

        # normalize it
        normalized_train_images = normalize_np_array(train_images)
        normalized_test_images = normalize_np_array(test_images)
        normalized_validation_images = normalize_np_array(validation_images)

        # store the normalized data in the data set
        data.train.set_images(normalized_train_images)
        data.test.set_images(normalized_test_images)
        data.validation.set_images(normalized_validation_images)

        # create the dataset holding the test, train and validation data
        # test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
        #                                          train_images, train_labels, 5000)
        #
        # return Datasets(train=train, validation=validation, test=test)
        #
        #
        # data.train.images = data.train.images._replace(v=node.v)

    return data


"""
Read .tfrecords
"""

# TODO:
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    return image


def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([FILE], num_epochs=1)
        image = read_and_decode(filename_queue)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while True:
                example = sess.run([image])
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def read_tfrecords(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                   reshape=False, seed=None, grey_scale=False):

    # instantiate a new protocol buffer, and fill in some of its fields.
    example = tf.train.Example()
    # TODO: remove hard coded filename
    for record in tf.python_io.tf_record_iterator("data/ship_train.tfrecords"):
        example.ParseFromString(record)
        f = example.features.feature
        v1 = f['int64 feature'].int64_list.value[0]
        v2 = f['float feature'].float_list.value[0]
        v3 = f['bytes feature'].bytes_list.value[0]
        # for bytes you might want to represent them in a different way (based on what they were before saving)
        # something like `np.fromstring(f['img'].bytes_list.value[0], dtype=np.uint8
        # Now do something with your v1/v2/v3


"""
Read pickle file
"""


def read_cifar10(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                 reshape=False, seed=None, grey_scale=False):
    """
    reads pickled cifar10 dataset
    :param data_dir: directory where the files are stored
    :param one_hot: whether labels should be stored as one hot vector
    :param num_classes: number of classes
    :param dtype:  dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param validation_size: size of the validation dataset
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :param grey_scale: load the images as gray scale
    :return:
    """

    # names of the cifar10 dataset
    cifar10_train_file_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
    cifar10_test_file_name = "data_batch_5"

    # holds the train images and labels
    train_images = []
    train_labels = []

    # iterate over the train filenames
    for filename in cifar10_train_file_names:
        # load the pickled file into a dictionary
        f = open(data_dir + '/cifar-10-batches-py/' + filename, 'rb')
        datadict = cPickle.load(f, encoding='latin1')
        f.close()
        # add the train images and the labels to the lists
        train_images.extend(datadict["data"])
        train_labels.extend(datadict['labels'])

    # load the test data file
    f = open(data_dir + '/cifar-10-batches-py/' + cifar10_test_file_name, 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()
    # store the test images and the labels in two numpy arrays
    test_images = np.array(datadict["data"])
    test_labels = np.array(datadict['labels'])

    # convert the lists to numpy array
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    if grey_scale:
        # grey scale luminosity conversion formula is 0.21 R + 0.72 G + 0.07 B
        test_images = test_images[:, :1024] * 0.21 + test_images[:, 1024:1024*2] * 0.72 + test_images[:, 1024*2:] * 0.07
        train_images = train_images[:, :1024] * 0.21 + train_images[:, 1024:1024*2] * 0.72 + train_images[:, 1024*2:] * 0.07

    # create the dataset holding the test, train and validation data
    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return Datasets(train=train, validation=validation, test=test)


"""
Read .mat file (svhn dataset) based on https://github.com/bdiesel/tensorflow-svhn/blob/master/svhn_data.py
"""


def convert_labels_to_one_hot(labels, num_classes):
    """
    converts the labels to one hot vectors
    :param labels: labels to convert
    :param num_classes: number of classes we have
    :return:
    """
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return labels


def reshape_image_array(img_array, grey_scale=False):
    """
    reshape the image array according to the color scale (rgb or grey scale)
    :param img_array: array of shape [rows, cols, color channels, number of images]
    :param grey_scale: whether we want to have grey scale images
    :return: the reshaped image array with shape [num_of_images, rows * cols * channels]
    """
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    channels = img_array.shape[2]
    num_imgs = img_array.shape[3]

    if grey_scale:
        channels = 1

    reshaped_img_array = np.empty(shape=(num_imgs, rows * cols * channels), dtype=np.float32)
    for x in range(0, num_imgs):

        if grey_scale:
            red_pixels = img_array[:, :, 0, x].flatten()
            reshaped_img_array[x] = red_pixels
        else:
            red_pixels = img_array[:, :, 0, x].flatten()
            green_pixels = img_array[:, :, 1, x].flatten()
            blue_pixels = img_array[:, :, 2, x].flatten()

            temp_rgb_array = np.append(red_pixels, green_pixels)
            temp_rgb_array = np.append(temp_rgb_array, blue_pixels)
            reshaped_img_array[x] = temp_rgb_array

    return reshaped_img_array


def read_mat_file(filename, one_hot=False, num_classes=10):
    """
    reads the mat file and returns the array holding the images
    :param filename: filename to read
    :param one_hot: whether labels should be returned as one hot vector
    :param num_classes: number of classes we have
    :return: array holding the image data with shape [num_of_images, rows * cols * channels], array holding the labels
    with shape [num_of_images, num_classes] if one_hot is true and shape [num_of_images] otherwise
    """
    data = loadmat(filename)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0  # Fix for weird labeling in dataset
    if one_hot:
        labels_one_hot = convert_labels_to_one_hot(labels, num_classes)
    else:
        labels_one_hot = labels
    # labels_one_hot = labels
    img_array = reshape_image_array(img_array=imgs)
    return img_array, labels_one_hot


def read_svhn_from_mat(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                       reshape=False, seed=None, grey_scale=False):
    """
    reads data from a .mat file and returns a DataSet object holding the train, test and validation data and labels
    :param data_dir: directory where the files are stored
    :param one_hot: whether labels should be stored as one hot vector
    :param num_classes: number of classes
    :param dtype:  dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param validation_size: size of the validation dataset
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :param grey_scale: load the images as grey scale
    :return:
    """

    train_filename = data_dir + "/svhn_train_32x32.mat"
    test_filename = data_dir + "/svhn_test_32x32.mat"

    # read the train file
    train_file = open(train_filename, 'rb')
    train_images, train_labels = read_mat_file(filename=train_file, one_hot=one_hot, num_classes=num_classes)
    train_file.close()

    # read the test file
    test_file = open(test_filename, 'rb')
    test_images, test_labels = read_mat_file(filename=test_file, one_hot=one_hot, num_classes=num_classes)
    test_file.close()

    if grey_scale:
        print("grey scale")

        # grey scale luminosity conversion formula is 0.21 R + 0.72 G + 0.07 B
        test_images = test_images[:, :1024] * 0.21 + test_images[:, 1024:1024*2] * 0.72 + test_images[:, 1024*2:] * 0.07
        train_images = train_images[:, :1024] * 0.21 + train_images[:, 1024:1024*2] * 0.72 + train_images[:, 1024*2:] * 0.07

    # create the dataset holding the test, train and validation data
    test, train, validation = create_dataset(dtype, num_classes, False, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return Datasets(train=train, validation=validation, test=test)


"""
Read ubyte files
"""


def get_bytes_from_file(filename):
    return open(filename, "rb").read()


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """ Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        f: A file object that can be passed into a gzip reader.

    Returns:
        data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
        ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.

    Returns:
        labels: a 1D uint8 numpy array.

    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


def read_mnist_data_from_ubyte(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, reshape=True,
                               validation_size=5000, seed=None):
    """
    reads mnist data from a ubyte file and returns a DataSet object holding the train, test and validation data and labels
    :param data_dir: directory where the files are stored
    :param one_hot: whether labels should be stored as one hot vector
    :param num_classes: number of classes
    :param dtype:  dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param validation_size: size of the validation dataset
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :return:
    """

    source_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, data_dir, source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, data_dir, source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, data_dir, source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, data_dir, source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

    test, train, validation = create_dataset(dtype, num_classes, False, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return Datasets(train=train, validation=validation, test=test)


"""
read csv files
"""


def read_image_and_labels_from_csv(filename, dim_x=28, dim_y=28):
    """
    reads image data and labels from a csv file; assumes labels is stored in the first column
    :param filename:
    :return:
    """
    data = pandas.read_csv(filename, sep=',', header=None)
    labels = data.ix[:, 0].values
    data = data.drop(data.columns[0], axis=1)   # drop the labels
    images = data.values

    n_data_points = labels.size
    images = images.reshape(n_data_points, dim_x, dim_y, 1)

    return images, labels


def read_csv_data_set(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                      reshape=True, seed=None):
    """
    reads a csv data set and returns a DataSet object holding the train, test and validation data and labels
    :param data_dir: directory where the files are stored
    :param one_hot: whether labels should be stored as one hot vector
    :param num_classes: number of classes
    :param dtype:  dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param validation_size: size of the validation dataset
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :return:
    """
    TRAIN_IMAGES = 'mnist_train.csv'
    TEST_IMAGES = 'mnist_test.csv'

    train_images, train_labels = read_image_and_labels_from_csv(data_dir + '/' + TRAIN_IMAGES)
    test_images, test_labels = read_image_and_labels_from_csv(data_dir + '/' + TEST_IMAGES)

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return Datasets(train=train, validation=validation, test=test)


"""
read image data (png)
"""


def read_images_from_dir(data_dir, n_classes=10, image_fileformat='.png', dim_x=28, dim_y=28, color_channels=1):
    """
    reads the images from the data directory
    :param data_dir: ../data/mnist_png/test_algorithm/
    :param n_classes: number of classes we have
    :param image_fileformat: fileformat of the image
    :param dim_x: x resolution of the image
    :param dim_y: y resolution of the image
    :param color_channels: number of channels (1=gray scale, 3=rgb)
    :return:
    """

    images = []
    labels = []
    n_data_points = 0

    for n in range(n_classes):
        for image_path in glob.glob(data_dir + str(n) + "/*" + image_fileformat):
            image = imageio.imread(image_path)
            images.append(image)
            labels.append(n)
            n_data_points += 1

    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(n_data_points, dim_x, dim_y, color_channels)

    return images, labels


def read_mnist_from_png(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                        reshape=True, seed=None):
    """
    reads the png images from the data directories and returns a DataSet object holding the train, test and validation
    data and labels
    :param data_dir: directory where the files are stored
    :param one_hot: whether labels should be stored as one hot vector
    :param num_classes: number of classes
    :param dtype:  dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param validation_size: size of the validation dataset
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :return:
    """
    print("reading test data")
    test_images, test_labels = read_images_from_dir(data_dir + "/mnist_png/test_algorithm/", image_fileformat="png")

    print("reading train data")
    train_images, train_labels = read_images_from_dir(data_dir + "/mnist_png/training/", image_fileformat="png")

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return Datasets(train=train, validation=validation, test=test)


def create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels, train_images, train_labels,
                   validation_size):
    """
    creates three datasets holding the test, train and the validation data and labels and returns them
    :param dtype: dtype` can be either 'uint8' to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]
    :param num_classes: number of classes we have
    :param one_hot: whether labels should be stored as one hot vector
    :param reshape: true: Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
    (assuming depth == 1)
    :param seed: for shuffling the data set
    :param test_images: numpy array holding the test images
    :param test_labels: numpy array holding the test labels
    :param train_images: numpy array holding the train images
    :param train_labels: numpy array holding the train labels
    :param validation_size: int; the desired size of our validation dataset
    :return: DataSet test, DataSet train, DataSet validation
    """
    if one_hot:
        train_labels = dense_to_one_hot(train_labels, num_classes)
        test_labels = dense_to_one_hot(test_labels, num_classes)
    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    options = dict(dtype=dtype, reshape=reshape)
    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    return test, train, validation

#################
# Mass spec data
#################


def create_mass_spec_spectrum_unidentified(single_entry):

    # create the iterator
    iterator = iter(single_entry)

    # skip the BEGIN IONS header and the empty row, if it exists
    if not next(iterator).startswith("BEGIN IONS"):
        next(iterator)

    title = next(iterator).split("=")[1].split(",")[0]
    pepmass = next(iterator).split("=")[1]
    charge = next(iterator).split("=")[1]
    sequence = next(iterator).split("=")[1]
    peaks = []

    next_row = next(iterator)
    while not next_row.startswith("END IONS"):
        peaks.append([float(a) for a in next_row.split(" ")])
        next_row = next(iterator)

    return {"title": title, "pepmass": pepmass, "charge": charge, "sequence": sequence, "peaks": peaks}


def load_mgf_file(filename):

    file_content = []

    with open(filename) as f:
        single_entry = []
        for line in f.readlines():

            single_entry.append(line.strip())
            if line.startswith("END IONS"):
                mass_spec_spectrum = create_mass_spec_spectrum_unidentified(single_entry)

                file_content.append(mass_spec_spectrum)
                single_entry = []

    return file_content


def create_mass_spec_spectrum_identified(single_entry):

    # create the iterator
    iterator = iter(single_entry)

    first_row = next(iterator)
    if not first_row.startswith("Name:"):
        first_row = next(iterator)

    title = first_row.split(" ")[1]
    sequence = title.split("/")[0]
    charge = title.split("/")[1]

    pepmass = next(iterator).split(" ")[1]
    comment = next(iterator).split(" ")[1]
    peaks = []

    # skip the num peaks row, if it exists
    next_row = next(iterator)
    if next_row.startswith("Num"):
        next(iterator)
    else:
        peaks.append(next_row.split(" "))

    # iterate over the remaining rows holding the peptide masses
    for row in iterator:
        # first if m/z, second is intensity
        try:
            la = [float(a) for a in row.split(" ")]
            if len(la) != 2:
                return None
            else:
                peaks.append(la)
        except ValueError:
            return None

    return {"title": title, "sequence": sequence, "charge": charge, "pepmass": pepmass, "comment": comment,
            "peaks": peaks}


def load_msp_file(filename):

    file_content = []

    with open(filename) as f:
        single_entry = []
        for line in f.readlines():
            if line == "\n" and len(single_entry) > 1:
                mass_spec_spectrum = create_mass_spec_spectrum_identified(single_entry)
                if mass_spec_spectrum:
                    file_content.append(mass_spec_spectrum)
                single_entry = []
            single_entry.append(line.strip())

    return file_content


def preprocess_mass_spec_data(mass_spec_data, n_peaks_to_keep=30):

    # get the peaks for all spectra
    peaks = [spectrum["peaks"] for spectrum in mass_spec_data]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_peaks(a, n_peaks_to_keep) for a in peaks]

    # remove None from the list
    filtered_peaks = [i for i in filtered_peaks if i is not None]

    peak_features = [create_features_for_peak(peak) for peak in filtered_peaks]

    return peak_features


def create_features_for_peak(peak):
    """
    represent each peak by 3 numbers (1) square root of its height (2) its location (mz distance from 0) and (3) its location relative
to the precursor (mz distance relative to precursor)
    """

    # (1) square root of its height
    intensities = peak[:, 1]

    try:
        intensities = np.sqrt(intensities)
    except TypeError:
        intensities = np.sqrt(intensities.astype(float))      # convert to float if necessary

    # (2) its location (mz distance from 0)
    mz_distance = peak[:, 0]

    # (3) its location relative to the precursor (mz distance relative to precursor)
    try:
        rel_distances = np.array([round(x - y, 2) for x, y in zip(mz_distance[1:], mz_distance)])
    except TypeError:
        mz_distance = mz_distance.astype(float)
        rel_distances = np.array([round(x - y, 2) for x, y in zip(mz_distance[1:], mz_distance)])

    return np.array([intensities, mz_distance, rel_distances])


def filter_peaks(peaks_to_filter, n_peaks_to_keep):

    # convert list to np.array
    peaks_to_filter = np.array(peaks_to_filter)

    if len(peaks_to_filter) < n_peaks_to_keep:
        return None

    # get the intensities
    intensities = peaks_to_filter[:, 1]

    # get the indices for the n highest intensities
    indices_to_keep = np.argsort(intensities)[:n_peaks_to_keep]

    # sort the indices, so the m/z values are in proper order
    indices_to_keep = np.sort(indices_to_keep)

    return peaks_to_filter[indices_to_keep]


def testing():
    """
    :return:
    """

    """
    read mass spec data
    """

    if True:

        # TODO: save preprocessed data to some file

        """
        Unidentified spectra
        """
        if True:

            print("#"*5 + " Unidentified spectra " + "#"*5)

            unidentified_spectra = load_mgf_file("../../data/mass_spec_data/results_saccharomyces_cerevisiae_unidentified.mgf")
            print(unidentified_spectra[15])

            peak_features_unidentified = preprocess_mass_spec_data(unidentified_spectra, 50)
            print(peak_features_unidentified[5])

            if True:

                num_peaks_unidentified_spectra = [len(spectrum["peaks"]) for spectrum in unidentified_spectra]
                print(len(num_peaks_unidentified_spectra))
                print(np.mean(num_peaks_unidentified_spectra))
                print(np.std(num_peaks_unidentified_spectra))
                print(np.min(num_peaks_unidentified_spectra))
                print(np.max(num_peaks_unidentified_spectra))

                la = [a for a in num_peaks_unidentified_spectra if a > 50]
                print(len(la))

                plt.hist(num_peaks_unidentified_spectra)
                plt.title("Num. peaks - unidentified spectra")
                plt.xlabel("Num. peaks")
                plt.show()

        """
        Identified spectra
        """

        if True:

            print("#"*5 + " Identified spectra " + "#"*5)
            identified_spectra = load_msp_file("../../data/mass_spec_data/Saccharomyces_cerevisiae_identified_fixed.txt")
            print(identified_spectra[15000])

            peak_features_identified = preprocess_mass_spec_data(identified_spectra, 50)
            print(peak_features_identified[5])

            if True:

                # some analysis regarding the number of peaks
                num_peaks_identified_spectra = [len(spectrum["peaks"]) for spectrum in identified_spectra]
                print(len(num_peaks_identified_spectra))
                print(np.mean(num_peaks_identified_spectra))
                print(np.std(num_peaks_identified_spectra))
                print(np.min(num_peaks_identified_spectra))
                print(np.max(num_peaks_identified_spectra))

                la = [a for a in num_peaks_identified_spectra if a > 50]
                print(len(la))

                plt.hist(num_peaks_identified_spectra)
                plt.title("Num. peaks - identified spectra")
                plt.xlabel("Num. peaks")
                plt.show()



    """
    read cifar10
    """

    if False:
        cifar10 = read_cifar10('../../data', one_hot=True)
        first_img, _ = cifar10.train.next_batch(1)

        print(_)

        lala = first_img.reshape(3, 32, 32)
        plt.imshow(np.transpose(lala, [1, 2, 0]))
        plt.show()

    """
    read .mat file
    """

    if False:

        grey_scale = False

        if grey_scale:
            start = timeit.default_timer()
            print("read .mat")

            svhn = read_svhn_from_mat('../../data', grey_scale=True, one_hot=True)
            first_img, _ = svhn.train.next_batch(1)

            print(_)

            stop = timeit.default_timer()
            print(stop - start)
            lala = first_img.reshape(32, 32)

            plt.gray()
            plt.imshow(lala)
            plt.show()
        else:
            start = timeit.default_timer()
            print("read .mat")

            svhn = read_svhn_from_mat('../../data', grey_scale=False, one_hot=True)
            first_img, _ = svhn.train.next_batch(1)

            print(_)

            print(first_img.shape)

            stop = timeit.default_timer()
            print(stop - start)

            # do some reshaping to display the images properly
            red = first_img[:, :1024].reshape(32, 32, 1)
            green = first_img[:, 1024:2048].reshape(32, 32, 1)
            blue = first_img[:, 2048:].reshape(32, 32, 1)

            plt.imshow(np.concatenate([red, green, blue], 2))
            plt.show()

    """
    read ubyte file
    """

    if False:
        start = timeit.default_timer()
        print("read ubyte")

        mnist = read_mnist_data_from_ubyte('../../data', one_hot=True)
        ubyte_first_img, _ = mnist.train.next_batch(1)

        print(_)

        stop = timeit.default_timer()
        print(stop - start)

        first_image = ubyte_first_img.reshape([28, 28])
        plt.gray()
        plt.imshow(first_image)
        plt.show()

    """
    read csv file: The format is: label, pix-11, pix-12, pix-13, ...
    """

    if False:

        start = timeit.default_timer()
        print("read csv")

        # test reading csv files
        train_images = read_csv_data_set('../../data', one_hot=True)
        csv_first_img, _ = train_images.train.next_batch(1)

        print(_)

        stop = timeit.default_timer()
        print(stop - start)

        first_image = csv_first_img.reshape([28, 28])
        plt.gray()
        plt.imshow(first_image)
        plt.show()

    """
    read png files:
    """

    if False:

        start = timeit.default_timer()
        print("read png")

        mnist = read_mnist_from_png('../../data', one_hot=True)
        png_first_img, _ = mnist.train.next_batch(1)

        print(_)

        stop = timeit.default_timer()
        print(stop - start)

        first_image = png_first_img.reshape([28, 28])
        plt.gray()
        plt.imshow(first_image)
        plt.show()


if __name__ == '__main__':
    testing()
