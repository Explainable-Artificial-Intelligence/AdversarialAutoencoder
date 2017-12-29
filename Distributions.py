import numpy as np
import matplotlib.pyplot as plt
import collections


def rotate_2D_array_by_angle(x_, y_, angle):
    """
    rotates the points in the 2D array by the specified angle
    :param x_: points in x dimension
    :param y_: points in y dimension
    :param angle: angle in degree
    :return: rotated x and y points
    """
    angle = np.deg2rad(angle)
    x_rot = x_ * np.cos(angle) - y_ * np.sin(angle)
    y_rot = x_ * np.sin(angle) + y_ * np.cos(angle)
    return x_rot, y_rot


class Distribution:
    def __init__(self, n_classes=1):
        """
        constructor
        :param n_classes: number of classes the distribution has
        """
        self.n_classes = n_classes
        self.samples = {}

    def create_class_samples(self, n_samples=5000):
        """
        creates the class samples and stores them in the samples dictionary
        :param n_samples: number of samples to create
        :return:
        """
        # create the distribution for each class and store it in the samples attribute

        n_samples_per_class = int(n_samples / self.n_classes) + 1
        for class_i in range(self.n_classes):
            x, y = self.create_class_specific_samples(class_i, n_samples_per_class)
            sample_tuple = collections.namedtuple("Sample", ["x", "y"])
            self.samples.update({"Class" + str(class_i): sample_tuple(x, y)})

    def create_class_specific_samples(self, class_i, n_samples_per_class=1000):
        """
        needs to be implemented by the classes inheriting
        :param class_i: class to create the samples for
        :param n_samples_per_class: number of samples for each class
        :return:
        """
        raise NotImplementedError

    def get_samples_for_class_i(self, class_i, n_samples=500):
        """
        returns n_samples for the specified class
        :param class_i: the class to draw the samples from
        :param n_samples: the number of samples to return
        :return:
        """

        x, y = self.create_class_specific_samples(class_i, n_samples)
        labels = [class_i] * n_samples

        return x, y, labels

    def get_samples(self, n_samples=500):
        """
        returns n_samples from all classes equally distributed
        :param n_samples: number of samples to return
        :return: x: numpy array of x_coords, y: numpy array of y_coords, labels: numpy array of class labels
        """

        if n_samples < self.n_classes:
            n_samples = self.n_classes

        n_samples_per_class = int(n_samples / self.n_classes) + 1

        # create the class samples
        self.create_class_samples(n_samples=n_samples)

        # iterate over the classes and add the respective coords and labels of the classes to the three lists
        x, y, labels = [], [], []
        for class_i in range(self.n_classes):
            sample_for_class_i = self.samples["Class" + str(class_i)]
            x = np.append(x, sample_for_class_i.x[:n_samples_per_class])
            y = np.append(y, sample_for_class_i.y[:n_samples_per_class])
            labels = np.append(labels, [class_i] * n_samples_per_class)

        # shuffle the points randomly
        perm0 = np.arange(n_samples)
        np.random.shuffle(perm0)
        x = x[perm0]
        y = y[perm0]
        labels = labels[perm0]

        return x, y, labels

    def plot_samples(self):
        """
        plots the distributions in a scatter plot
        :return:
        """
        if not self.samples:
            self.create_class_samples()
        for class_id in self.samples:
            sample = self.samples[class_id]
            plt.scatter(sample.x, sample.y)
        plt.show()


class FlowerDistribution(Distribution):
    def __init__(self, n_classes=10, mean_x=0.0, mean_y=5.0, var_x=0.5, var_y=10):
        super().__init__(n_classes)

        # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multivariate_normal.html
        self.mean = [mean_x, mean_y]
        self.cov = [[var_x, 0], [0, var_y]]

    def create_class_specific_samples(self, class_i, n_samples_per_class=1000):
        rotation_angle = 360 / self.n_classes * class_i
        x, y = np.random.multivariate_normal(self.mean, self.cov, n_samples_per_class).T
        x_rot, y_rot = rotate_2D_array_by_angle(x_=x, y_=y, angle=rotation_angle)
        return x_rot, y_rot


class SwissRollDistribution(Distribution):
    def __init__(self, n_classes=10, spread=1.5, noise=0.0):
        super().__init__(n_classes)
        self.spread = spread
        self.noise = noise

    def create_class_specific_samples(self, class_i, n_samples_per_class=1000):
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

        temp = np.arange(0, 2, 1/self.n_classes)
        t = 3 * self.spread * np.pi * (temp[class_i] + 0.1 * generator.rand(1, n_samples_per_class))
        # t = self.spread * np.pi * (1 + 2 * generator.rand(1, self.n_samples))

        x = t * np.cos(t)
        y = t * np.sin(t)

        X = np.concatenate((x, y))
        X += self.noise * generator.randn(2, n_samples_per_class)
        X = X.T

        x = X[:, 0]
        y = X[:, 1]

        return x, y


class GaussianDistribution(Distribution):
    def __init__(self, n_classes=10, mean_x=0.0, mean_y=5.0, var_x=0.5, var_y=10):
        super().__init__(n_classes)

        # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multivariate_normal.html
        self.mean = [mean_x, mean_y]
        self.cov = [[var_x, 0], [0, var_y]]

    def create_class_specific_samples(self, class_i, n_samples_per_class=1000):
        x, y = np.random.multivariate_normal(self.mean, self.cov, n_samples_per_class).T
        return x, y


def testing():
    """
    """

    """
    Test flower distribution
    """
    var_y = 10
    mean_y = var_y / 2 + 10
    flowers = FlowerDistribution(n_classes=10, mean_x=0, mean_y=mean_y, var_x=0.1, var_y=var_y)
    flowers.plot_samples()
    print(flowers.get_samples_for_class_i(class_i=2, n_samples=5))
    print(flowers.get_samples_for_class_i(class_i=2, n_samples=5))
    print(flowers.get_samples(5947))

    """
    Test swiss roll distribution
    """
    swiss_roll = SwissRollDistribution(n_classes=10, spread=1.5, noise=0.3)
    swiss_roll.plot_samples()
    print(swiss_roll.get_samples_for_class_i(class_i=1, n_samples=10))

    """
    Test gaussian distribution
    """
    gaussian = GaussianDistribution(n_classes=10, mean_x=0, mean_y=0, var_x=10, var_y=10)
    gaussian.create_class_samples(100)
    gaussian.plot_samples()
    gaussian.create_class_samples(10000)
    gaussian.plot_samples()
    print(gaussian.get_samples(50))


if __name__ == '__main__':
    testing()
