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
    def __init__(self, n_classes=10, n_samples=5000):
        """
        constructor
        :param n_classes: number of classes the distribution has
        :param n_samples: number of samples the distribution should provide
        """
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.distributions = {}
        self.class_sample = collections.namedtuple("Sample", ["x", "y"])

    def create_class_distributions(self):
        """
        creates the class distributions and stores them in the distributions dictionary
        :return:
        """
        # create the distriubution for each class and store it in the distributions attribute
        for class_i in range(self.n_classes):
            x, y = self.create_class_specific_distribution(class_i)
            self.distributions.update({"Class" + str(class_i): self.class_sample(x, y)})

    def create_class_specific_distribution(self, class_i):
        """
        needs to be implemented by the child classes
        :param class_i: the class to create the distribution for
        :return:
        """
        raise NotImplementedError

    def get_n_samples_for_class_i(self, class_i, n_samples=500):
        """
        returns n_samples for the specified class
        :param class_i: the class to draw the samples from
        :param n_samples: the number of samples to return
        :return:
        """
        if not self.distributions:
            self.create_class_distributions()
        sample_for_class = self.distributions["Class" + str(class_i)]
        x = sample_for_class.x
        y = sample_for_class.y
        # TODO: iterator instead of returning same proportion of the array
        # TODO: recreate distribution, if there not enough data points
        return x[:n_samples], y[:n_samples]

    def plot_distributions(self):
        """
        plots the distributions in a scatter plot
        :return:
        """
        if not self.distributions:
            self.create_class_distributions()
        for class_id in self.distributions:
            sample = self.distributions[class_id]
            plt.scatter(sample.x, sample.y)
        plt.show()

    def get_n_samples(self, n_samples=500):
        """
        returns n_samples from all classes equally distributed
        :param n_samples:
        :return:
        """

        if not self.distributions:
            self.create_class_distributions()

        x = []
        y = []
        n_samples_per_class = int(n_samples / self.n_classes)

        for class_i in range(self.n_classes):

            sample_for_class = self.distributions["Class" + str(class_i)]

            # TODO: iterator instead of returning same proportion of the array
            x += sample_for_class.x[:n_samples_per_class]
            y += sample_for_class.y[:n_samples_per_class]

        return x, y


class FlowerDistribution(Distribution):

    def __init__(self, n_classes=10, n_samples=5000, mean_x=0.0, mean_y=5.0, var_x=0.5, var_y=10):

        super().__init__(n_classes, n_samples)

        # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multivariate_normal.html
        self.mean = [mean_x, mean_y]
        self.cov = [[var_x, 0], [0, var_y]]

    def create_class_specific_distribution(self, class_i):
        rotation_angle = 360 / self.n_classes * class_i
        x, y = np.random.multivariate_normal(self.mean, self.cov, self.n_samples).T
        x_rot, y_rot = rotate_2D_array_by_angle(x_=x, y_=y, angle=rotation_angle)
        return x_rot, y_rot


class SwissRollDistribution(Distribution):

    def __init__(self, n_classes=10, n_samples=5000, spread=1.5, noise=0.0):

        super().__init__(n_classes, n_samples)
        self.spread = spread
        self.noise = noise

    def create_class_specific_distribution(self, class_i):
        """
        :param class_i:
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

        # TODO: include classes

        generator = np.random.RandomState(5)

        t = self.spread * np.pi * (1 + 2 * generator.rand(1, self.n_samples))
        x = t * np.cos(t)
        z = t * np.sin(t)

        X = np.concatenate((x, z))
        X += self.noise * generator.randn(2, self.n_samples)
        X = X.T

        x = X[:,0]
        y = X[:,1]

        return x, y


class GaussianDistribution(Distribution):

    def __init__(self, n_classes=10, n_samples=5000, mean_x=0.0, mean_y=5.0, var_x=0.5, var_y=10):

        super().__init__(n_classes, n_samples)

        # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multivariate_normal.html
        self.mean = [mean_x, mean_y]
        self.cov = [[var_x, 0], [0, var_y]]

    def create_class_specific_distribution(self, class_i):
        x, y = np.random.multivariate_normal(self.mean, self.cov, self.n_samples).T
        return x, y


def testing():
    """
    """
    # TODO: mean_x and mean_y for flower distribution should probably be dependend on var_x and var_y
    # var_y = 5; mean_y = 10

    """
    Test flower distribution
    """
    var_y = 10
    mean_y = var_y/2+10
    flowers = FlowerDistribution(n_classes=10, n_samples=1000, mean_x=0, mean_y=mean_y, var_x=0.1, var_y=var_y)
    flowers.plot_distributions()
    print(flowers.get_n_samples_for_class_i(class_i=2, n_samples=5))

    """
    Test swiss roll distribution
    """
    swiss_roll = SwissRollDistribution(n_classes=10, n_samples=10000, spread=1.5, noise=0.1)
    print(swiss_roll.get_n_samples_for_class_i(class_i=1, n_samples=10))
    swiss_roll.plot_distributions()

    """
    Test gaussian distribution
    """
    gaussian = GaussianDistribution(n_classes=10, n_samples=1000, mean_x=0, mean_y=0, var_x=10, var_y=10)
    gaussian.plot_distributions()


if __name__ == '__main__':
    testing()




