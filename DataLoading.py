import gzip

import imageio as imageio
import numpy as np
import pandas
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat
from six.moves import cPickle

import timeit

"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""

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
    # TODO: implement grey scale

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

    # create the dataset holding the test, train and validation data
    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return base.Datasets(train=train, validation=validation, test=test)


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


def read_mat_file(filename, one_hot=False, num_classes=10, grey_scale=False):
    """
    reads the mat file and returns the array holding the images
    :param filename: filename to read
    :param one_hot: whether labels should be returned as one hot vector
    :param num_classes: number of classes we have
    :param grey_scale: whether we want to have grey scale images
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
    img_array = reshape_image_array(img_array=imgs, grey_scale=grey_scale)
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
    train_images, train_labels = read_mat_file(filename=train_file, one_hot=one_hot, num_classes=num_classes,
                                               grey_scale=grey_scale)
    train_file.close()

    # read the test file
    test_file = open(test_filename, 'rb')
    test_images, test_labels = read_mat_file(filename=test_file, one_hot=one_hot, num_classes=num_classes,
                                             grey_scale=grey_scale)
    test_file.close()

    # create the dataset holding the test, train and validation data
    test, train, validation = create_dataset(dtype, num_classes, False, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return base.Datasets(train=train, validation=validation, test=test)


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

    return base.Datasets(train=train, validation=validation, test=test)


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

    return base.Datasets(train=train, validation=validation, test=test)


"""
read image data (png)
"""


def read_images_from_dir(data_dir, n_classes=10, image_fileformat='.png', dim_x=28, dim_y=28, color_channels=1):
    """
    reads the images from the data directory
    :param data_dir: ./data/mnist_png/testing/
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
    test_images, test_labels = read_images_from_dir(data_dir + "/mnist_png/testing/", image_fileformat="png")

    print("reading train data")
    train_images, train_labels = read_images_from_dir(data_dir + "/mnist_png/training/", image_fileformat="png")

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return base.Datasets(train=train, validation=validation, test=test)


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
    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    return test, train, validation


def testing():
    """
    :return:
    """

    """
    read cifar10
    """

    if False:
        cifar10 = read_cifar10('./data', one_hot=True)
        first_img, _ = cifar10.train.next_batch(1)

        print(_)

        lala = first_img.reshape(3, 32, 32)
        plt.imshow(np.transpose(lala, [1, 2, 0]))
        plt.show()

    """
    read .mat file
    """

    if True:

        grey_scale = False

        if grey_scale:
            start = timeit.default_timer()
            print("read .mat")

            svhn = read_svhn_from_mat('./data', grey_scale=True, one_hot=True)
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

            svhn = read_svhn_from_mat('./data', grey_scale=False, one_hot=True)
            first_img, _ = svhn.train.next_batch(1)

            print(_)

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

        mnist = read_mnist_data_from_ubyte('./data', one_hot=True)
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
        train_images = read_csv_data_set('./data', one_hot=True)
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

        mnist = read_mnist_from_png('./data', one_hot=True)
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
