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
import tensorflow as tf
from scipy.io import loadmat
from six.moves import cPickle

import timeit

"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

"""
Read tfrecords file
"""


# TODO: implement


def read_and_decode(filename_queue, input_dim):
    """
    reads the filename queue and returns the images and labels
    :param filename_queue: filename queue to read
    :param input_dim: input dimension
    :return: image and label
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            # TODO:
            # 'image_raw': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    # TODO:
    # image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.decode_raw(features['image'], tf.uint8)
    # TODO:
    # image.set_shape([mnist.IMAGE_PIXELS])
    image.set_shape([input_dim])
    # image.set_shape([IMAGE_SIZE*IMAGE_SIZE])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.    Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert from [0, 255] -> [0, 1.0] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(filename, batch_size, input_dim):
    """Reads input data.

    Args:
        filename: File name to read as input
        batch_size: Number of examples per returned batch.
        input_dim: input dimension of the data

    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
            in the range [0, 1.0].
        * labels is an int32 tensor with shape [batch_size] with the true label,
            a number in the range [0, mnist.NUM_CLASSES).
    """
    # filename = os.path.join(self.FLAGS.data_dir, self.TRAIN_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue, input_dim)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return images, sparse_labels


def read_cifar10(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                 reshape=False, seed=None, one_channel=False):
    cifar10_train_file_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
    cifar10_test_file_name = "data_batch_5"

    train_images = []
    train_labels = []
    for filename in cifar10_train_file_names:
        f = open(data_dir + '/cifar-10-batches-py/' + filename, 'rb')
        datadict = cPickle.load(f, encoding='latin1')
        f.close()
        X = datadict["data"]
        Y = datadict['labels']

        train_images.extend(X)
        train_labels.extend(Y)

    f = open(data_dir + '/cifar-10-batches-py/' + cifar10_test_file_name, 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    print(train_images.shape)

    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    test_images = np.array(datadict["data"])
    test_labels = np.array(datadict['labels'])

    # TODO: check if reshape is needed

    # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    # Y = np.array(Y)
    #
    # # Visualizing CIFAR 10
    # fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    # for j in range(5):
    #     for k in range(5):
    #         i = np.random.choice(range(len(X)))
    #         axes1[j][k].set_axis_off()
    #         axes1[j][k].imshow(X[i:i + 1][0])
    #
    # plt.show()

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

    return base.Datasets(train=train, validation=validation, test=test)



"""
Read .mat file (svhn dataset) based on https://github.com/bdiesel/tensorflow-svhn/blob/master/svhn_data.py
"""


def convert_labels_to_one_hot(labels, num_classes):
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return labels


def convert_imgs_to_array(img_array, pixel_depth=255, one_channel=False):
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    chans = img_array.shape[2]
    num_imgs = img_array.shape[3]
    scalar = 1 / pixel_depth
    # Note: not the most efficent way but can monitor what is happening
    # new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    # for x in range(0, num_imgs):
    #     new_array[x] = img_array[:, :, :, x]
    # return new_array.reshape(num_imgs, 32*32*3)

    if one_channel:
        chans = 1

    new_array = np.empty(shape=(num_imgs, rows * cols * chans), dtype=np.float32)
    for x in range(0, num_imgs):

        if one_channel:
            red_pixels = img_array[:, :, 0, x].flatten()
            new_array[x] = red_pixels
        else:
            red_pixels = img_array[:, :, 0, x].flatten()
            green_pixels = img_array[:, :, 1, x].flatten()
            blue_pixels = img_array[:, :, 2, x].flatten()

            temp = np.append(red_pixels, green_pixels)
            temp = np.append(temp, blue_pixels)
            new_array[x] = temp


            # print(img_array[:, :, :, x].shape)
            # print(img_array[:, :, :, x])
            # tf.concat([red_pixels_input_images, green_pixels_input_images, blue_pixels_input_images], 3)
            # np.stack(red_pixels, green_pixels, blue_pixels, axis=3)
            # new_array[x] = np.concatenate(red_pixels, green_pixels, blue_pixels, axis=3)

            # TODO: works
            # new_array[x] = img_array[:, :, :, x].flatten()

            # new_array[x] = img_array[:, :, :, x]
    return new_array


def process_svhn_file(file, one_hot=False, num_classes=10, one_channel=False):
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0  # Fix for weird labeling in dataset
    if one_hot:
        labels_one_hot = convert_labels_to_one_hot(labels, num_classes)
    else:
        labels_one_hot = labels
    img_array = convert_imgs_to_array(img_array=imgs, one_channel=one_channel)
    return img_array, labels_one_hot


def read_svhn_from_mat(data_dir, one_hot=False, num_classes=10, dtype=dtypes.float32, validation_size=5000,
                       reshape=False, seed=None, one_channel=False):
    train_filename = data_dir + "/svhn_train_32x32.mat"
    test_filename = data_dir + "/svhn_test_32x32.mat"

    train_file = open(train_filename, 'rb')
    train_images, train_labels = process_svhn_file(file=train_file, one_hot=one_hot, num_classes=num_classes,
                                                   one_channel=one_channel)
    train_file.close()

    test_file = open(test_filename, 'rb')
    test_images, test_labels = process_svhn_file(file=test_file, one_hot=one_hot, num_classes=num_classes,
                                                 one_channel=one_channel)
    test_file.close()

    # TODO:

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


"""
Read other file formats
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


def read_mnist_data_from_ubyte(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True,
                               validation_size=5000,
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

    train_images, train_labels = read_image_and_labels(train_dir + '/' + TRAIN_IMAGES)
    test_images, test_labels = read_image_and_labels(train_dir + '/' + TEST_IMAGES)

    test, train, validation = create_dataset(dtype, num_classes, one_hot, reshape, seed, test_images, test_labels,
                                             train_images, train_labels, validation_size)

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
    read cifar10
    """

    cifar10 = read_cifar10('./data', one_hot=True)
    first_img, _ = cifar10.train.next_batch(1)

    print(first_img.shape)
    lala = first_img.reshape(3, 32, 32)
    plt.imshow(np.transpose(lala, [1, 2, 0]))
    plt.show()

    return

    """
    read .mat file
    """

    start = timeit.default_timer()
    print("read .mat")

    svhn = read_svhn_from_mat('./data', one_channel=True)
    first_img, _ = svhn.train.next_batch(1)

    stop = timeit.default_timer()
    print(stop - start)
    lala = first_img.reshape(32, 32)

    # for 3 channels
    # lala = first_img.reshape(32, 32, 3)

    # for one channel
    plt.gray()
    plt.imshow(lala)

    # for 3 channels:
    # plt.imshow(np.transpose(lala, [1,2,0]))
    plt.show()

    return

    """
    read ubyte file
    """

    start = timeit.default_timer()
    print("read ubyte")

    mnist = read_mnist_data_from_ubyte('./data', one_hot=True)
    ubyte_first_img, _ = mnist.train.next_batch(1)

    stop = timeit.default_timer()
    print(stop - start)

    first_image = ubyte_first_img.reshape([28, 28])
    plt.gray()
    plt.imshow(first_image)
    plt.show()

    """
    read csv file: The format is: label, pix-11, pix-12, pix-13, ...
    """

    start = timeit.default_timer()
    print("read csv")

    # test reading csv files
    train_images = read_csv_data_set('./data', one_hot=True)
    csv_first_img, _ = train_images.train.next_batch(1)

    stop = timeit.default_timer()
    print(stop - start)

    first_image = csv_first_img.reshape([28, 28])
    plt.gray()
    plt.imshow(first_image)
    plt.show()

    """
    read png files:
    """

    start = timeit.default_timer()
    print("read png")

    mnist = read_data_in_images('./data', one_hot=True)
    png_first_img, _ = mnist.train.next_batch(1)

    stop = timeit.default_timer()
    print(stop - start)

    first_image = ubyte_first_img.reshape([28, 28])
    plt.gray()
    plt.imshow(first_image)
    plt.show()


if __name__ == '__main__':
    testing()
