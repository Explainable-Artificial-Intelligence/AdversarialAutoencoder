import random

import numpy as np
import matplotlib.pyplot as plt
from math import *


def draw_from_np_distribution(n_samples=1, **distribution_parameters):
    """
    wrapper function to draw from the distributions implemented in numpy.random.
    :param n_samples: int or tuple of ints, Output shape.
    :param distribution_parameters: dictionary holding
        required:
            - distribution_name: the distribution to draw from
            - return_type: ['float', 'int']: type of the value this function should return
        optionally:
            - distribution specific parameters: e.g. the lower and upper boundary for the uniform distribution.
            Note: necessary if it's required to draw from the distribution!
            - is_positive: returned value should be >= 0
            - is_greater_than_zero: returned value should be > 0
            - is_negative: returned value should be <= 0
            - is_smaller_than_zero: returned value should be < 0
    :return:
    """

    # set n_samples to None, so the np.random functions return a single value
    if n_samples == 1:
        n_samples = None

    return_value = None

    if distribution_parameters["distribution_name"] == "beta":
        # a: alpha, float, non-negative; b: beta, float, non-negative
        return_value = np.random.beta(a=distribution_parameters["a"], b=distribution_parameters["b"],
                                      size=n_samples)
    elif distribution_parameters["distribution_name"] == "binomial":
        # n: int, >= 0; p: float, 0<=p<=1
        return_value = np.random.binomial(n=distribution_parameters["n"], p=distribution_parameters["p"],
                                          size=n_samples)
    elif distribution_parameters["distribution_name"] == "chisquare":
        # df: degrees of freedom, int
        return_value = np.random.chisquare(df=distribution_parameters["df"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "dirichlet":
        # alpha: array, e.g. (10, 5, 3)
        return_value = np.random.dirichlet(alpha=distribution_parameters["alpha"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "exponential":
        # scale: float
        return_value = np.random.exponential(scale=distribution_parameters["scale"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "f":
        # dfnum: int, >0; dfden, int, >0
        return_value = np.random.f(dfnum=distribution_parameters["dfnum"], dfden=distribution_parameters["dfden"],
                                   size=n_samples)
    elif distribution_parameters["distribution_name"] == "gamma":
        # shape: float, >0; scale: float, >0, default=1
        return_value = np.random.gamma(shape=distribution_parameters["shape"],
                                       scale=distribution_parameters["scale"],
                                       size=n_samples)
    elif distribution_parameters["distribution_name"] == "geometric":
        # p: float (probability)
        return_value = np.random.geometric(p=distribution_parameters["p"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "gumbel":
        # loc: float, default=0; scale: float, default=1
        return_value = np.random.gumbel(loc=distribution_parameters["loc"], scale=distribution_parameters["scale"],
                                        size=n_samples)
    elif distribution_parameters["distribution_name"] == "hypergeometric":
        # ngood: int, >=0; nbad: int, >=0; nsample: int, >=1, <= ngood+nbad
        return_value = np.random.hypergeometric(ngood=distribution_parameters["ngood"],
                                                nbad=distribution_parameters["nbad"],
                                                nsample=distribution_parameters["nsample"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "laplace":
        # loc: float, default=0; scale: float, default=1
        return_value = np.random.laplace(loc=distribution_parameters["loc"], scale=distribution_parameters["scale"],
                                         size=n_samples)
    elif distribution_parameters["distribution_name"] == "logistic":
        # loc: float, default=0; scale: float, default=1
        return_value = np.random.logistic(loc=distribution_parameters["loc"],
                                          scale=distribution_parameters["scale"],
                                          size=n_samples)
    elif distribution_parameters["distribution_name"] == "lognormal":
        # mean: float, default=0; sigma: >0, default=1
        return_value = np.random.lognormal(mean=distribution_parameters["mean"],
                                           sigma=distribution_parameters["sigma"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "logseries":
        # p: float, 0<p<1
        return_value = np.random.logseries(p=distribution_parameters["p"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "multinomial":
        # n: int; pvals: array of floats, sum(pvals)=1
        return_value = np.random.multinomial(n=distribution_parameters["n"], pvals=distribution_parameters["pvals"],
                                             size=n_samples)
    elif distribution_parameters["distribution_name"] == "multivariate_normal":
        # mean: 1D array with length N; cov: 2D array of shape (N,N); tol: float
        return_value = np.random.multivariate_normal(mean=distribution_parameters["mean"],
                                                     cov=distribution_parameters["cov"],
                                                     tol=distribution_parameters["tol"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "negative_binomial":
        # n: int, >0; p: float, 0<=p<=1;
        return_value = np.random.negative_binomial(n=distribution_parameters["n"],
                                                   p=distribution_parameters["p"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "noncentral_chisquare":
        # df: int, >0; nonc: float, >0
        return_value = np.random.noncentral_chisquare(df=distribution_parameters["df"],
                                                      nonc=distribution_parameters["nonc"],
                                                      size=n_samples)
    elif distribution_parameters["distribution_name"] == "noncentral_f":
        # dfnum: int, >1; dfden: int, >1; nonc: float, >=0
        return_value = np.random.noncentral_f(dfnum=distribution_parameters["dfnum"],
                                              dfden=distribution_parameters["dfden"],
                                              nonc=distribution_parameters["nonc"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "normal":
        # loc: float; scale: float
        return_value = np.random.normal(loc=distribution_parameters["loc"], scale=distribution_parameters["scale"],
                                        size=n_samples)
    elif distribution_parameters["distribution_name"] == "pareto":
        # a: float, >0
        return_value = np.random.pareto(a=distribution_parameters["a"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "poisson":
        # lam: float, >=0
        return_value = np.random.poisson(lam=distribution_parameters["lam"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "power":
        # a: float, >=0
        return_value = np.random.power(a=distribution_parameters["a"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "rayleigh":
        # scale: float, >=0, default=1
        return_value = np.random.rayleigh(scale=distribution_parameters["scale"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "standard_cauchy":
        return_value = np.random.standard_cauchy(size=n_samples)
    elif distribution_parameters["distribution_name"] == "standard_gamma":
        # shape: float, >0
        return_value = np.random.standard_gamma(shape=distribution_parameters["shape"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "standard_normal":
        return_value = np.random.standard_normal(size=n_samples)
    elif distribution_parameters["distribution_name"] == "standard_t":
        # df: int, >0
        return_value = np.random.standard_t(df=distribution_parameters["df"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "triangular":
        # left: float; mode: float, left <= mode <= right; right: float, >left
        return_value = np.random.triangular(left=distribution_parameters["left"],
                                            mode=distribution_parameters["mode"],
                                            right=distribution_parameters["right"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "uniform":
        # low: float, default=0; high: float, default=1
        return_value = np.random.uniform(low=distribution_parameters["low"], high=distribution_parameters["high"],
                                         size=n_samples)
    elif distribution_parameters["distribution_name"] == "vonmises":
        # mu: float; kappa: float, >=0
        return_value = np.random.vonmises(mu=distribution_parameters["mu"], kappa=distribution_parameters["kappa"],
                                          size=n_samples)
    elif distribution_parameters["distribution_name"] == "wald":
        # mean: float, >0; scale: float, >=0
        return_value = np.random.wald(mean=distribution_parameters["mean"], scale=distribution_parameters["scale"],
                                      size=n_samples)
    elif distribution_parameters["distribution_name"] == "weibull":
        # a: float, >0
        return_value = np.random.weibull(a=distribution_parameters["a"], size=n_samples)
    elif distribution_parameters["distribution_name"] == "zipf":
        # a: float, >1
        return_value = np.random.zipf(a=distribution_parameters["a"], size=n_samples)

    # TODO: works only for n_samples=1 -> make it work on higher values as well

    # output should be >= 0
    if distribution_parameters.get("is_positive"):
        return_value = abs(return_value)
    # output should be > 0
    if distribution_parameters.get("is_greater_than_zero"):
        if distribution_parameters["return_type"] == "int":
            return_value = abs(return_value) + 1
        else:
            return_value = abs(return_value) + 1e-12
    # output should be <= 0
    if distribution_parameters.get("is_negative"):
        return_value = abs(return_value) * -1
    # output should be < 0
    if distribution_parameters.get("is_smaller_than_zero"):
        if distribution_parameters["return_type"] == "int":
            return_value = abs(return_value) * -1 - 1
        else:
            return_value = abs(return_value) * -1 - 1e-12

    # change type of the return value
    if distribution_parameters.get("return_type"):
        if distribution_parameters["return_type"] == "float":
            return_value = float(return_value)
        elif distribution_parameters["return_type"] == "int":
            return_value = int(return_value)
    else:
        pass

    return return_value


def draw_from_dim_reduced_dataset(n_samples, dataset, z_dim):
    """
    draws from the dataset which has gotten their dimension reduced
    :param n_samples: number of samples to draw
    :param dataset: ["MNIST", "SVHN", "cifar10"]
    :param z_dim: dimensionality
    :return:
    """

    # restore np array from numpy file
    # TODO: no more hard coded file names
    data = np.load('../data/dimension_reduced_data/' + dataset + '/pca/SVHN_pca_z_dim_8' + '.npy')

    # restore labels
    # data_labels = np.load('../data/dimension_reduced_data/mnist_z_dim_equals_2_labels.npy')

    noise_distribution = {"distribution_name": "normal", "loc": 0, "scale": 0.1}

    # try it with some noise
    random_data = select_randomly_from_np_array(data, n_samples, noise_distribution)

    return random_data


def select_randomly_from_np_array_with_label(np_array, np_array_labels, n_samples, int_label, noise_distribution_params):
    """
    returns n_samples samples with label=int_label randomly selected from the numpy array
    :param np_array: numpy array to select from
    :param np_array_labels: numpy array of the labels
    :param n_samples: how many samples to return
    :param int_label: desired label of the samples
    :param noise_distribution_params: dictionary holding
        required:
            - distribution_name: the distribution to draw from
        optionally:
            - distribution specific parameters: e.g. the lower and upper boundary for the uniform distribution.
            Note: necessary if it's required to draw from the distribution!
            - is_positive: returned value should be >= 0
            - is_greater_than_zero: returned value should be > 0
            - is_negative: returned value should be <= 0
            - is_smaller_than_zero: returned value should be < 0
    :return:
    """
    samples_for_label = np_array[np.where(np_array_labels == int_label)]
    random_indices = np.random.choice(samples_for_label.shape[0], n_samples, replace=False)

    random_samples = samples_for_label[random_indices]

    if noise_distribution_params:
        noise_array = draw_from_np_distribution(n_samples=random_samples.shape, **noise_distribution_params)
        # apply noise
        random_samples += noise_array

    return random_samples


def select_randomly_from_np_array(np_array, n_samples, noise_distribution_params):
    """
    returns n_samples samples randomly selected from the numpy array
    :param np_array: numpy array to select from
    :param n_samples: how many samples to return
    :param noise_distribution_params: dictionary holding
        required:
            - distribution_name: the distribution to draw from
        optionally:
            - distribution specific parameters: e.g. the lower and upper boundary for the uniform distribution.
            Note: necessary if it's required to draw from the distribution!
            - is_positive: returned value should be >= 0
            - is_greater_than_zero: returned value should be > 0
            - is_negative: returned value should be <= 0
            - is_smaller_than_zero: returned value should be < 0
    :return:
    """
    random_indices = np.random.choice(np_array.shape[0], n_samples, replace=False)
    random_samples = np_array[random_indices]

    if noise_distribution_params:
        noise_array = draw_from_np_distribution(n_samples=random_samples.shape, **noise_distribution_params)
        # apply noise
        random_samples += noise_array

    return random_samples


def create_class_specific_samples_swiss_roll(class_i: int, n_classes: int, spread: float, noise: float,
                                             n_samples_per_class: int = 1000, no_noise: bool = False) -> np.array:
    """
    :param class_i: class to createt the samples for
    :param n_samples_per_class: number of samples to create
    :return:
    Based on:
        http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
    The algorithm is from Marsland [1].
    References
    ----------
    .. [1] S. Marsland, "Machine Learning: An Algorithmic Perspective",
           Chapter 10, 2009.
           http://seat.massey.ac.nz/personal/s.r.marsland/Code/10/lle.py
    """

    generator = np.random.RandomState()

    temp = np.arange(0, 2, 1 / n_classes)

    if no_noise:
        t = [3 * spread * np.pi * (temp[class_i] + 0.1 * np.array([0.5]*n_samples_per_class))]
        # t = self.spread * np.pi * (1 + 2 * generator.rand(1, self.n_samples))
    else:
        t = 3 * spread * np.pi * (temp[class_i] + 0.1 * generator.rand(1, n_samples_per_class))
        # t = self.spread * np.pi * (1 + 2 * generator.rand(1, self.n_samples))

    # print(t)
    # print(t.shape)
    #
    # t = 3 * spread * np.pi * (temp[class_i] + 0.1 * generator.rand(1, n_samples_per_class))
    # print(t)
    # print(t.shape)


    x = t * np.cos(t)
    y = t * np.sin(t)

    X = np.concatenate((x, y))
    X += noise * generator.randn(2, n_samples_per_class)
    X = X.T

    x = list(X[:, 0])
    y = list(X[:, 1])

    return x, y


def draw_from_swiss_roll(n_classes=10, spread=1.5, noise=0.0, shape=(100, 2), shuffle=True):
    """
    draws from a swiss roll distribution
    :param n_classes:
    :param spread: spread of the swiss role
    :param noise: standard deviation of the gaussian noise.
    :param shape: shape of the generated data
    :return: [(x1, y1), (x2, y2), .., (xn, yn)]
    """

    if shape[1] != 2:
        raise NotImplementedError

    # TODO: implement
    if n_classes != 10:
        raise NotImplementedError

    # number of points we want at max for each class
    n_samples = shape[0]
    n_samples_per_class = int(math.ceil(n_samples / n_classes))

    # holds the generated x and y points
    generated_points_x = np.array([])
    generated_points_y = np.array([])

    # generate the points and app them to the arrays
    for class_i in range(n_classes):
        x, y = create_class_specific_samples_swiss_roll(class_i=class_i, n_classes=n_classes, spread=spread, noise=noise,
                                                        n_samples_per_class=n_samples_per_class, no_noise=False) # TODO: set no noise to False
        generated_points_x = np.append(generated_points_x, x)
        generated_points_y = np.append(generated_points_y, y)

    # shuffle the samples
    if shuffle:
        perm0 = np.arange(n_samples)
        np.random.shuffle(perm0)
        generated_points_x = generated_points_x[perm0]
        generated_points_y = generated_points_y[perm0]

    # truncate the arrays
    generated_points_x = generated_points_x[:n_samples]
    generated_points_y = generated_points_y[:n_samples]

    return np.column_stack((generated_points_x, generated_points_y))


def draw_from_single_gaussian(mean=0.0, std_dev=1.0, shape=(100, 2)):
    """
    draws random samples from a normal gaussian distribution
    :param mean: mean of the distribution
    :param std_dev: standard deviation of the distribution
    :param shape: output shape. If shape is None a single value is returned.
    :return:
    """
    return np.random.normal(loc=mean, scale=std_dev, size=shape)


def points_on_circumference(center=(0, 0), radius=10, n=100):
    """
    returns n points laying on the circumference of the circle with the provided center and radius.
    Based on: https://gist.github.com/danleyb2/ce6d2b82b1556f7bb7dc3c5d2bccb2fc
    :param center: center of the circle
    :param radius: radius of the circle
    :param n: number of points to return
    :return: [(x1, y1), (x2, y2), .., (xn, yn)]
    """

    return [(center[0] + (math.cos(2 * math.pi / n * x) * radius),
             center[1] + (math.sin(2 * math.pi / n * x) * radius)) for x in range(0, n + 1)]


def draw_from_multiple_gaussians(n_classes=10, sigma=1, shape=(100, 2), shuffle=True):

    # TODO: parameters for circle radii and centers
    # TODO: drawn only from std normal distributions -> add it to the function description somewhere

    # number of points we want at max for each class
    n_samples = shape[0]
    n_samples_per_class = math.ceil(n_samples / n_classes)
    z_dim = shape[1]
    shape_for_class = (n_samples_per_class, z_dim)

    # gaussians should be on the circumference on a circle, so we need to modify the mean accordingly
    mu_for_different_gaussians = points_on_circumference(center=(0, 0), radius=10, n=n_classes)

    # holds the generated points
    generated_samples = []

    # plot the different classes on the latent space
    for class_label, mu in zip(range(n_classes), mu_for_different_gaussians):

        # TODO: make it work for z_dim; currently to prevent crashing if z_dim > 2
        if z_dim > 2:
            mu = mu + (z_dim - 2) * (0,)

        # get the points corresponding to the same classes
        points_for_current_class_label = np.random.normal(mu, sigma, shape_for_class)

        # add them to the list
        generated_samples.extend(list(points_for_current_class_label))

    # convert to np array for reshuffling the samples
    generated_samples = np.array(generated_samples)

    # shuffle the samples
    if shuffle:
        perm0 = np.arange(shape[0])
        np.random.shuffle(perm0)
        generated_samples = generated_samples[perm0]

    # truncate the samples
    generated_samples = generated_samples[:shape[0]]

    return np.array(generated_samples).reshape(shape)


def walk_along_multiple_gaussians(center=(0, 0), radius=10, n_points_to_return=100):
    """
    function "walks" along the centers of multiple gaussians laying on the circumference of a circle and returns
    the provided number of points as np.array
    :param center: center of the circle
    :param radius: radius of the circle
    :param n_points_to_return: number of points to return
    :return:
    """
    return np.array(points_on_circumference(center=center, radius=radius, n=n_points_to_return))


# def walk_along_swiss_roll():
#
#     # TODO: draw from swiss roll without random shuffling and get every nth item
#
#     n_classes = 10
#     shape = (1000, 2)
#     spread = 1.5
#     noise = 0.0
#
#     # number of points we want at max for each class
#     n_samples = shape[0]
#     n_samples_per_class = int(math.ceil(n_samples / n_classes))
#
#     # holds the generated x and y points
#     generated_points_x = np.array([])
#     generated_points_y = np.array([])
#
#     # generate the points and app them to the arrays
#     for class_i in range(n_classes):
#         x, y = create_class_specific_samples_swiss_roll(class_i=class_i, n_classes=n_classes, spread=spread, noise=noise,
#                                                         n_samples_per_class=n_samples_per_class, no_noise=True)
#         generated_points_x = np.append(generated_points_x, x)
#         generated_points_y = np.append(generated_points_y, y)


def walk_along_single_gaussian(min_value=-10, max_value=10, n_points_to_return=(14, 14)):
    """
    walk along a single gaussian by laying a grid over it and sampling this grid evenly.
    :param min_value:
    :param max_value: max_value of the gaussian
    :param n_points_to_return:
    :return:
    """

    # TODO: parameter for z_dim; sample randomly instead of evenly

    # number fo points
    x_n = n_points_to_return[0]
    y_n = n_points_to_return[1]

    x_points_spacing = abs(max_value-min_value) / x_n
    y_points_spacing = abs(max_value-min_value) / y_n

    # creates evenly spaced values within [-10, 10] with a spacing of 1.5
    x_points = np.arange(max_value, min_value, -x_points_spacing).astype(np.float32)
    y_points = np.arange(min_value, max_value, y_points_spacing).astype(np.float32)

    nx, ny = len(x_points), len(y_points)

    lala = []

    # iterate over the image grid
    for i in range(nx*ny):
        # create a data point from the x_points and y_points array as input for the decoder
        z = np.concatenate(([y_points[int(i % nx)]], [x_points[int(i / ny)]]))
        # z = np.reshape(z, (1, 2))

        lala.append(z)

    print(lala)
    print(np.array(lala))
    return lala


# TODO: taken from: https://github.com/musyoku/adversarial-autoencoder/blob/master/aae/sampler.py
def sample_gaussian_mixture(x, y, label, num_labels):
    shift = 1.4
    r = 2.0 * np.pi / float(num_labels) * float(label)
    new_x = x * cos(r) - y * sin(r)
    new_y = x * sin(r) + y * cos(r)
    new_x += shift * cos(r)
    new_y += shift * sin(r)
    return np.array([new_x, new_y]).reshape((2,))


def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample_gaussian_mixture(x[batch, zi], y[batch, zi],
                                                            random.randint(0, num_labels - 1), num_labels)
    return z*5


def supervised_gaussian_mixture(batchsize, ndim, label_indices, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample_gaussian_mixture(x[batch, zi], y[batch, zi], label_indices[batch],
                                                            num_labels)
    return z*5


def sample_swiss_roll(label, num_labels):
    uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
    r = sqrt(uni) * 3.0
    rad = np.pi * 4.0 * sqrt(uni)
    x = r * cos(rad)
    y = r * sin(rad)
    return np.array([x, y]).reshape((2,))


def sample_swiss_roll_modified(label, num_labels, random_value):
    uni = random_value / float(num_labels) + float(label) / float(num_labels)
    r = sqrt(uni) * 3.0
    rad = np.pi * 4.0 * sqrt(uni)
    x = r * cos(rad)
    y = r * sin(rad)
    return np.array([x, y]).reshape((2,))


def swiss_roll(batchsize, ndim, num_labels):
    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample_swiss_roll(random.randint(0, num_labels - 1), num_labels)
    return z*5


def supervised_swiss_roll(batchsize, ndim, label_indices, num_labels):
    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample_swiss_roll(label_indices[batch], num_labels)
    return z*5


def walk_along_gaussian_mixture(n_points_per_class, n_classes=10):
    samples = []
    step_size = 1 / n_points_per_class
    for j in range(0, n_classes):
        for i in np.arange(-1.2, 1.2, step_size):
            z = sample_gaussian_mixture(i, i/50, j, n_classes)*5
            samples.append(z)

    return np.array(samples)


def walk_along_swiss_roll(n_points_per_class, n_classes=10):
    samples = []
    step_size = 1 / n_points_per_class
    for j in range(0, n_classes):
        for i in np.arange(0, 1, step_size):
            z = sample_swiss_roll_modified(j, n_classes, i)*5
            samples.append(z)

    return np.array(samples)


def visualize_walk_along_gaussian_mixture(n_points_per_class, n_classes=10):
    # try default
    batch_size = 10000
    z_dim = 2
    batch_labels_int = np.random.choice(n_classes, batch_size)
    z_real_dist_labeled = supervised_gaussian_mixture(batch_size, z_dim, batch_labels_int, n_classes)
    plt.scatter(z_real_dist_labeled[:, 0], z_real_dist_labeled[:, 1])

    step_size = 1 / n_points_per_class
    for j in range(0, n_classes):
        for i in np.arange(-1.2, 1.2, step_size):
            z = sample_gaussian_mixture(i, i/50, j, n_classes)*5
            plt.scatter(z[0], z[1])

    plt.show()


def visualize_walk_along_swiss_roll(n_points_per_class, n_classes=10):
    # try default
    batch_size = 10000
    z_dim = 2
    batch_labels_int = np.random.choice(n_classes, batch_size)
    z_real_dist_labeled = supervised_swiss_roll(batch_size, z_dim, batch_labels_int, n_classes)
    plt.scatter(z_real_dist_labeled[:, 0], z_real_dist_labeled[:, 1])

    step_size = 1 / n_points_per_class
    for j in range(0, n_classes):
        for i in np.arange(0, 1, 0.1):
            z = sample_swiss_roll_modified(j, n_classes, i)*5
            plt.scatter(z[0], z[1])

    plt.show()


