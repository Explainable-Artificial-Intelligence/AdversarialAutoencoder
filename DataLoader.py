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

import timeit

"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

"""

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


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


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, validation_size=5000,
                   seed=None, source_url=DEFAULT_SOURCE_URL):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir, source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir, source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, train_dir, source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir, source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

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

    return base.Datasets(train=train, validation=validation, test=test)


def read_image_and_labels(filename):

    data = pandas.read_csv(filename, sep=',', header=None)
    labels = data.ix[:, 0].values
    data = data.drop(data.columns[0], axis=1)
    images = data.values

    n_data_points = labels.size

    # TODO: change from hard coded
    images = images.reshape(n_data_points, 28, 28, 1)

    return images, labels


def read_csv_data_set(train_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                      reshape=True, seed=None):
    TRAIN_IMAGES = 'mnist_train.csv'
    TEST_IMAGES = 'mnist_test.csv'

    start = timeit.default_timer()

    train_images, train_labels = read_image_and_labels(train_dir + '/' + TRAIN_IMAGES)
    test_images, test_labels = read_image_and_labels(train_dir + '/' + TEST_IMAGES)

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    stop = timeit.default_timer()

    print("creating dataset")
    print(stop - start)

    return base.Datasets(train=train, validation=validation, test=test)


def read_images_from_dir(image_dir, n_classes=10, image_fileformat='.png'):
    """

    :param image_dir: ./data/mnist_png/testing/
    :param n_classes:
    :param image_fileformat:
    :return:
    """

    images = []
    labels = []
    n_data_points = 0

    for n in range(n_classes):
        for image_path in glob.glob(image_dir + str(n) + "/*" + image_fileformat):
            image = imageio.imread(image_path)
            images.append(image)
            labels.append(n)
            n_data_points += 1

    images = np.array(images)
    labels = np.array(labels)
    # TODO: change from hard coded
    images = images.reshape(n_data_points, 28, 28, 1)

    return images, labels


def read_data_in_images(train_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                        reshape=True, seed=None):

    print("reading test data")
    test_images, test_labels = read_images_from_dir(train_dir + "/mnist_png/testing/", image_fileformat="png")

    print("reading train data")
    train_images, train_labels = read_images_from_dir(train_dir + "/mnist_png/training/", image_fileformat="png")

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return base.Datasets(train=train, validation=validation, test=test)


def create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels, train_images, train_labels,
                   validation_size):
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
    read ubyte file
    """

    start = timeit.default_timer()

    mnist = read_data_sets('./data', one_hot=True)
    ubyte_first_img, _ = mnist.train.next_batch(1)
    # print(mnist.train.images)
    # print(type(mnist.train.images))
    # print(mnist.train.images[0])

    # first_image = ubyte_first_img.reshape([28, 28])
    # plt.gray()
    # plt.imshow(first_image)
    # plt.show()

    stop = timeit.default_timer()

    print(stop - start)

    """
    read csv file: The format is: label, pix-11, pix-12, pix-13, ...
    """

    # start = timeit.default_timer()

    # test reading csv files
    train_images = read_csv_data_set('./data', one_hot=True)
    csv_first_img, _ = train_images.train.next_batch(1)

    first_image = csv_first_img.reshape([28, 28])
    plt.gray()
    plt.imshow(first_image)
    plt.show()

    # stop = timeit.default_timer()

    print(stop - start)
    """
    read png files:
    """
    # mnist = read_data_in_images('./data', one_hot=True)
    # png_first_img, _ = mnist.train.next_batch(1)
    #
    # first_image = ubyte_first_img.reshape([28, 28])
    # plt.gray()
    # plt.imshow(first_image)
    # plt.show()


if __name__ == '__main__':
    testing()
