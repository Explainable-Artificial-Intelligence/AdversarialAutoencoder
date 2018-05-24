"""
    Holds the functions shared by all three Autoencoders (Unsupervised, Supervised and SemiSupervised).
"""
import glob
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from sklearn.decomposition import PCA

from swagger_server.utils.Storage import Storage
from util.NeuralNetworkUtils import get_all_layer_vars


def reshape_tensor_to_rgb_image(image_array, input_dim_x, input_dim_y):
    """
    reshapes the given image array to an rgb image for the tensorboard visualization
    :param image_array: array of images to be reshaped
    :param input_dim_x: dim x of the images
    :param input_dim_y: dim y of the images
    :return: rgb array of the image
    """
    n_colored_pixels_per_channel = input_dim_x * input_dim_y
    red_pixels = tf.reshape(image_array[:, :n_colored_pixels_per_channel],
                            [-1, input_dim_x, input_dim_y, 1])
    green_pixels = tf.reshape(image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2],
                              [-1, input_dim_x, input_dim_y, 1])
    blue_pixels = tf.reshape(image_array[:, n_colored_pixels_per_channel * 2:],
                             [-1, input_dim_x, input_dim_y, 1])
    rgb_image = tf.concat([red_pixels, green_pixels, blue_pixels], 3)
    return rgb_image


def reshape_image_array(aae_class, img_array, is_array_of_arrays=False):
    """
    reshapes the image array based on the color scale:
        - gray scale: [input_dim_x, input_dim_y]
        - rgb scale: [input_dim_x, input_dim_y, 3]
    :param aae_class: AAE instance to access some important fields from it
    :param img_array: array/list of the image
    :param is_array_of_arrays: whether we have an array of arrays or not
    :return: reshaped np.array
    """

    # reshape the images according to the color scale
    if aae_class.color_scale == "gray_scale":
        # matplotlib wants a 2D array
        img = np.array(img_array).reshape(aae_class.input_dim_x, aae_class.input_dim_y)
    else:
        image_array = np.array(img_array)
        n_colored_pixels_per_channel = aae_class.input_dim_x * aae_class.input_dim_y

        if is_array_of_arrays:
            # first n_colored_pixels_per_channel encode red
            red_pixels = image_array[:, :n_colored_pixels_per_channel].reshape(aae_class.input_dim_x,
                                                                               aae_class.input_dim_y, 1)
            # next n_colored_pixels_per_channel encode green
            green_pixels = image_array[:, n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                .reshape(aae_class.input_dim_x, aae_class.input_dim_y, 1)
            # last n_colored_pixels_per_channel encode blue
            blue_pixels = image_array[:, n_colored_pixels_per_channel * 2:].reshape(aae_class.input_dim_x,
                                                                                    aae_class.input_dim_y, 1)
        else:
            # first n_colored_pixels_per_channel encode red
            red_pixels = image_array[:n_colored_pixels_per_channel].reshape(aae_class.input_dim_x,
                                                                            aae_class.input_dim_y,
                                                                            1)
            # next n_colored_pixels_per_channel encode green
            green_pixels = image_array[n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2] \
                .reshape(aae_class.input_dim_x, aae_class.input_dim_y, 1)
            # last n_colored_pixels_per_channel encode blue
            blue_pixels = image_array[n_colored_pixels_per_channel * 2:].reshape(
                aae_class.input_dim_x,
                aae_class.input_dim_y, 1)

        # concatenate the color arrays into one array
        img = np.concatenate([red_pixels, green_pixels, blue_pixels], 2)

    return img


def draw_class_distribution_on_latent_space(latent_representations_current_epoch, labels_current_epoch, result_path,
                                            epoch, random_points_for_image_grid, combined_plot=False):
    """
    draws the class distribution on the latent space.
    :param latent_representations_current_epoch: list of shape (n_batches*batch_size, z_dim) holding the encoder output,
    -> the latent representation of the inputs
    :param labels_current_epoch: list of shape (n_batches*batch_size, n_classes), holds the labels of the inputs
    encoded as one-hot vectors
    :param result_path: path where to store the resulting image
    :param epoch: current epoch; for the file name of the resulting image
    :param random_points_for_image_grid: random points used for generating the image grid (only for (semi-) supervised
    autoencoders)
    :param combined_plot: whether the plot should be combined with the image grid
    :return:
    """

    if combined_plot:
        plt.subplot(1, 2, 2)

    # convert lists to numpy array
    latent_representations_current_epoch = np.array(latent_representations_current_epoch)
    labels_current_epoch = np.array(labels_current_epoch)

    # convert one hot vectors to integer labels
    int_labels = np.argmax(labels_current_epoch, axis=1)

    # get the dimension of the latent space and the number of classes
    z_dim = latent_representations_current_epoch.shape[1]
    n_classes = labels_current_epoch.shape[1]

    pca = PCA(n_components=2)

    # perform PCA if the dimension of the latent space is higher than 2
    if z_dim > 2:
        mu = np.mean(latent_representations_current_epoch, axis=0)

        pca.fit(latent_representations_current_epoch)
        latent_representations_current_epoch = pca.transform(latent_representations_current_epoch)

        plt.xlabel("Principal component 1: " + "{:6.4f}".format(pca.explained_variance_ratio_[0]))
        plt.ylabel("Principal component 2: " + "{:6.4f}".format(pca.explained_variance_ratio_[1]))

        # TODO: whats this for?
        Xhat = np.dot(latent_representations_current_epoch[:, :2], pca.components_[:2, :])
        Xhat += mu
        print()
        print(Xhat.shape)
        print(Xhat[0])

    # plot the different classes on the latent space
    for class_label in range(n_classes):
        # get the points corresponding to the same classes
        points_for_current_class_label = latent_representations_current_epoch[np.where(int_labels == class_label)]
        # plot them
        plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1], label=str(class_label))

    # draw the random points on the latent space if we have any
    if random_points_for_image_grid is not None:

        # markers for the random points
        markers = ["^", "v", ">", "<", "s", "*", "_", "+", "x", "|"]

        if z_dim > 2:
            random_points_for_image_grid = pca.transform(random_points_for_image_grid)

        for random_point, marker in zip(random_points_for_image_grid, markers):
            plt.scatter(random_point[0], random_point[1], marker=marker, c="black")

    if combined_plot:
        plt.suptitle("Epoch: " + str(epoch))

    plt.legend()
    plt.savefig(result_path + str(epoch) + "_latent_space_class_distribution" + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_epoch_summary_image(aae_class, epoch, include_tuning_performance=False):
    """
    creates a summary image displaying the losses and the learning rates over time, the real distribution and the latent
    representation, the discriminator outputs (pos and neg) and one input image and its reconstruction image
    :param aae_class: AAE instance to access some important fields from it
    :param epoch: current epoch of the training (only used for the image filename)
    :param include_tuning_performance: whether to include the losses and learning rates from the other adversarial
    autoencoders in the same tuning process in the plots
    :return:
    """

    epoch_summary_vars = aae_class.get_epoch_summary_vars()

    # convert lists to numpy array
    latent_representation = np.array(epoch_summary_vars["latent_representation"])
    batch_labels = np.array(epoch_summary_vars["batch_labels"])
    real_img = np.array(epoch_summary_vars["batch_x"])[0, :]    # get the first input image
    created_img = np.array(epoch_summary_vars["reconstructed_images"])[0, :] # get the reconstruction of it
    real_dist = np.array(epoch_summary_vars["real_dist"])
    discriminator_neg = np.array(epoch_summary_vars["discriminator_neg"])
    discriminator_pos = np.array(epoch_summary_vars["discriminator_pos"])

    # convert one hot vectors to integer labels
    batch_integer_labels = np.argmax(batch_labels, axis=1)

    # get the number of classes
    n_classes = batch_labels.shape[1]

    # calculate the total losses
    total_losses = [sum(x) for x in zip(aae_class.performance_over_time["autoencoder_losses"],
                                        aae_class.performance_over_time["discriminator_losses"],
                                        aae_class.performance_over_time["generator_losses"])]

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=4, ncols=3)

    """
    plot the losses over time
    """
    plt.subplot(4, 3, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["autoencoder_losses"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder loss")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 2)
    plt.stackplot(aae_class.performance_over_time["list_of_epochs"],
                  [aae_class.performance_over_time["discriminator_losses"],
                   aae_class.performance_over_time["generator_losses"]], labels=["discriminator loss",
                                                                                 "generator loss"])
    plt.title("discriminator-generator ratio")
    plt.legend()
    plt.xlabel("epoch")

    plt.subplot(4, 3, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"], total_losses)
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            # calculate the total losses
            total_loss_current_aae = [sum(x) for x in zip(performance_dict["autoencoder_losses"],
                                                          performance_dict["discriminator_losses"],
                                                          performance_dict["generator_losses"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot], total_loss_current_aae[:n_points_to_plot],
                     alpha=0.5, c="gray")
    plt.title("total loss")
    plt.xlabel("epoch")

    # usually we use a stack plot for the discriminator and generator loss, but this won't work for several results,
    # so we use our currently unused subplot to plot the losses in a line chart
    if include_tuning_performance:
        plt.subplot(4, 3, 4)
        discriminator_generator_loss = [sum(x) for x in zip(aae_class.performance_over_time["discriminator_losses"],
                                                            aae_class.performance_over_time["generator_losses"])]
        plt.plot(aae_class.performance_over_time["list_of_epochs"], discriminator_generator_loss)
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            # calculate the total losses
            discriminator_generator_loss_current_aae = [sum(x) for x in zip(performance_dict["discriminator_losses"],
                                                                            performance_dict["generator_losses"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     discriminator_generator_loss_current_aae[:n_points_to_plot], alpha=0.5, c="gray")
        plt.title("generator + discriminator loss")
        plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """
    # plot one input image..
    plt.subplot(4, 3, 5)
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(4, 3, 6)
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(4, 3, 7)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    plt.subplot(4, 3, 8)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
    else:
        # plot the different classes on the latent space
        for class_label in range(n_classes):
            # get the points corresponding to the same classes
            points_for_current_class_label = latent_representation[np.where(batch_integer_labels == class_label)]
            # plot them
            plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1],
                        label=str(class_label))
    plt.legend()
    plt.title("encoder dist")

    # plot the discriminator outputs
    plt.subplot(4, 3, 9)
    plt.hist(discriminator_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.legend()
    plt.title("discriminator distr.")

    """
    plot the learning rates over time
    """
    plt.subplot(4, 3, 10)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["autoencoder_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 11)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["discriminator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("discriminator_lr")
    plt.xlabel("epoch")

    plt.subplot(4, 3, 12)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["generator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_lr")
    plt.xlabel("epoch")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_epoch_summary_image_semi_supervised(aae_class, epoch, include_tuning_performance=False):
    """
    creates a summary image displaying the losses and the learning rates over time, the real distribution and the latent
    representation, the discriminator outputs (pos and neg) and one input image and its reconstruction image
    :param aae_class: AAE instance to access some important fields from it
    :param epoch: current epoch of the training (only used for the image filename)
    :param include_tuning_performance: whether to include the losses and learning rates from the other adversarial
    autoencoders in the same tuning process in the plots
    :return:
    """

    epoch_summary_vars = aae_class.get_epoch_summary_vars()

    # convert lists to numpy array
    latent_representation = np.array(epoch_summary_vars["latent_representation"])
    batch_labels = np.array(epoch_summary_vars["batch_labels"])
    real_img = np.array(epoch_summary_vars["batch_X_unlabeled"])[0, :]
    created_img = np.array(epoch_summary_vars["reconstructed_images"])[0, :]
    real_dist = np.array(epoch_summary_vars["real_dist"])
    discriminator_gaussian_neg = np.array(epoch_summary_vars["discriminator_gaussian_neg"])
    discriminator_gaussian_pos = np.array(epoch_summary_vars["discriminator_gaussian_pos"])

    # convert one hot vectors to integer labels
    batch_integer_labels = np.argmax(epoch_summary_vars["batch_labels"], axis=1)
    real_cat_dist_integer_labels = np.argmax(epoch_summary_vars["real_cat_dist"], axis=1)
    encoder_cat_dist_integer_labels = np.argmax(epoch_summary_vars["encoder_cat_dist"], axis=1)

    # get the number of classes
    n_classes = batch_labels.shape[1]

    # calculate the total losses
    total_losses = [sum(x) for x in zip(aae_class.performance_over_time["autoencoder_losses"],
                                        aae_class.performance_over_time["discriminator_gaussian_losses"],
                                        aae_class.performance_over_time["discriminator_categorical_losses"],
                                        aae_class.performance_over_time["generator_losses"],
                                        aae_class.performance_over_time["supervised_encoder_loss"])]

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=5, ncols=4)

    """
    plot the losses over time
    """
    plt.subplot(5, 4, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            plt.plot(performance_dict["list_of_epochs"], performance_dict["autoencoder_losses"], alpha=0.5, c="gray")
    plt.title("autoencoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 2)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_gaussian_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_gaussian_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_gaussian_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_categorical_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_categorical_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_categorical_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 4)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["generator_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["generator_losses"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 5)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["supervised_encoder_loss"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["supervised_encoder_loss"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("supervised_encoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 6)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             total_losses)
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            total_loss_current_aae = [sum(x) for x in zip(performance_dict["autoencoder_losses"],
                                                          performance_dict["discriminator_gaussian_losses"],
                                                          performance_dict["discriminator_categorical_losses"],
                                                          performance_dict["generator_losses"],
                                                          performance_dict["supervised_encoder_loss"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     total_loss_current_aae[:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("total loss")
    plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """

    # # plot one input image..
    plt.subplot(5, 4, 7)
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(5, 4, 8)
    # if the aae is not trained enough/has unfavorable parameters, it's possible that the reconstruction can hold
    # pixels with negative value
    if (created_img < 0).any():
        created_img = np.abs(created_img)
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the accuracy
    """
    if len(aae_class.performance_over_time["accuracy"]) > 0:  # for the first epoch we don't not have accuracy yet
        plt.subplot(5, 4, 9)
        plt.plot(aae_class.performance_over_time["accuracy_epochs"],
                 aae_class.performance_over_time["accuracy"])
        # plot the performance of the other adv autoencoders
        if include_tuning_performance:
            # get the dictionary holding the dictionaries which hold the performance over time
            tuning_performances = Storage.get_tuning_results_performance_over_time()
            for key, performance_dict in tuning_performances.items():
                n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
                plt.plot(performance_dict["accuracy_epochs"][:n_points_to_plot],
                         performance_dict["accuracy"][:n_points_to_plot], alpha=0.5, c="gray")
        plt.title("accuracy")
        plt.xlabel("epoch")

    """
    plot the batch labels and the predicted labels
    """
    plt.subplot(5, 4, 10)
    plt.hist(batch_integer_labels, alpha=0.5, label="batch labels", color="#d95f02")
    plt.hist(encoder_cat_dist_integer_labels, alpha=0.5, label="predicted labels", color="#1b9e77")
    plt.legend()
    plt.title("label prediction")

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(5, 4, 11)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    plt.subplot(5, 4, 12)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
    else:
        # plot the different classes on the latent space
        for class_label in range(n_classes):
            # get the points corresponding to the same classes
            points_for_current_class_label = latent_representation[np.where(batch_integer_labels == class_label)]
            # plot them
            plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1],
                        label=str(class_label))
    plt.legend()
    plt.title("encoder dist")

    """
    plot the discriminator outputs
    """
    plt.subplot(5, 4, 13)
    plt.hist(discriminator_gaussian_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_gaussian_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator gaussian")
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.hist(discriminator_gaussian_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_gaussian_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator categorical")
    plt.legend()

    """
    plot the learning rates over time
    """
    plt.subplot(5, 4, 16)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["autoencoder_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 17)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_g_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["discriminator_g_lr"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_g_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 18)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_c_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"], learning_rate_dict["discriminator_c_lr"], alpha=0.5,
                     c="gray")
    plt.title("discriminator_c_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 19)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["generator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 20)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["supervised_encoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["supervised_encoder_lr"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("supervised_encoder_lr")
    plt.xlabel("epoch")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def visualize_cluster_heads(aae_class, cluster_heads, epoch, mini_batch_i):

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=6, ncols=3)     # TODO: depending on n_clusters

    for i, cluster_head in enumerate(cluster_heads):
        plt.subplot(6, 3, i+1)

        img = reshape_image_array(aae_class, cluster_head)
        if aae_class.color_scale == "gray_scale":
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title("cluster head: " + str(i))

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '_cluster_heads.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_minibatch_summary_image_unsupervised_clustering(aae_class, real_dist, latent_representation, batch_x, decoder_output,
                                                   epoch, mini_batch_i, real_cat_dist, encoder_cat_dist, batch_labels,
                                                   discriminator_gaussian_neg, discriminator_gaussian_pos,
                                                   discriminator_cat_neg, discriminator_cat_pos,
                                                   include_tuning_performance=False):
    """
    creates a summary image displaying the losses and the learning rates over time, the real distribution and the latent
    representation, the discriminator outputs (pos and neg) and one input image and its reconstruction image
    :param aae_class: AAE instance to access some important fields from it
    :param real_dist: real distribution the AAE should map to
    :param latent_representation: latent representation of the AAE
    :param batch_x: current batch of input images
    :param decoder_output: output of the decoder for the current batch
    :param epoch: current epoch of the training (only used for the image filename)
    :param mini_batch_i: current iteration of the minibatch (only used for the image filename)
    :param real_cat_dist: real categorical distribution
    :param encoder_cat_dist: encoder output of the categorical distribution
    :param batch_labels: labels of the current batch
    :param discriminator_gaussian_neg: output of the gaussian discriminator for the negative samples q(z) (generated by
    the generator)
    :param discriminator_gaussian_pos: output of the gaussian discriminator  for the positive samples p(z) (from real
    data distribution)
    :param discriminator_cat_neg: output of the categorical discriminator for the negative samples
    :param discriminator_cat_pos: output of the categorical discriminator for the positive samples
    :param include_tuning_performance: whether to include the losses and learning rates from the other adversarial
    autoencoders in the same tuning process in the plots
    :return:
    """

    # TODO:

    epoch_summary_vars = aae_class.get_epoch_summary_vars()

    real_dist = epoch_summary_vars["real_dist"]


    # convert lists to numpy array
    latent_representation = np.array(latent_representation)
    batch_labels = np.array(batch_labels)

    # convert one hot vectors to integer labels
    batch_integer_labels = np.argmax(batch_labels, axis=1)
    real_cat_dist_integer_labels = np.argmax(real_cat_dist, axis=1)
    encoder_cat_dist_integer_labels = np.argmax(encoder_cat_dist, axis=1)

    # get the number of classes
    n_classes = batch_labels.shape[1]

    # calculate the total losses
    total_losses = [sum(x) for x in zip(aae_class.performance_over_time["autoencoder_losses"],
                                        aae_class.performance_over_time["discriminator_gaussian_losses"],
                                        aae_class.performance_over_time["discriminator_categorical_losses"],
                                        aae_class.performance_over_time["generator_losses"])]

    # increase figure size
    plt.rcParams["figure.figsize"] = (15, 20)

    # create the subplots
    plt.subplots(nrows=5, ncols=4)

    """
    plot the losses over time
    """
    plt.subplot(5, 4, 1)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["autoencoder_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            plt.plot(performance_dict["list_of_epochs"], performance_dict["autoencoder_losses"], alpha=0.5, c="gray")
    plt.title("autoencoder_loss")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 2)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_gaussian_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_gaussian_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_gaussian_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 3)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["discriminator_categorical_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["discriminator_categorical_losses"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_categorical_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 4)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             aae_class.performance_over_time["generator_losses"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     performance_dict["generator_losses"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_losses")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 6)
    plt.plot(aae_class.performance_over_time["list_of_epochs"],
             total_losses)
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which hold the performance over time
        tuning_performances = Storage.get_tuning_results_performance_over_time()
        for key, performance_dict in tuning_performances.items():
            n_points_to_plot = len([i for i in performance_dict["list_of_epochs"] if i <= epoch + 1])
            total_loss_current_aae = [sum(x) for x in zip(performance_dict["autoencoder_losses"],
                                                          performance_dict["discriminator_gaussian_losses"],
                                                          performance_dict["discriminator_categorical_losses"],
                                                          performance_dict["generator_losses"])]
            plt.plot(performance_dict["list_of_epochs"][:n_points_to_plot],
                     total_loss_current_aae[:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("total loss")
    plt.xlabel("epoch")

    """
    plot one input image and its reconstruction
    """

    # # plot one input image..
    plt.subplot(5, 4, 7)
    real_img = batch_x[0, :]
    img = reshape_image_array(aae_class, real_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # .. and its reconstruction
    plt.subplot(5, 4, 8)
    created_img = decoder_output[0, :]
    # if the aae is not trained enough/has unfavorable parameters, it's possible that the reconstruction can hold
    # pixels with negative value
    if (created_img < 0).any():
        created_img = np.abs(created_img)
    img = reshape_image_array(aae_class, created_img)
    if aae_class.color_scale == "gray_scale":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    """
    plot the batch labels and the predicted labels
    """
    plt.subplot(5, 4, 10)
    plt.hist(batch_integer_labels, alpha=0.5, label="batch labels", color="#d95f02")
    plt.hist(encoder_cat_dist_integer_labels, alpha=0.5, label="predicted labels", color="#1b9e77")
    plt.legend()
    plt.title("label prediction")

    """
    plot the encoder and the real gaussian distribution
    """
    # plot the real distribution
    plt.subplot(5, 4, 11)
    if aae_class.z_dim > 2:
        plt.hist(real_dist.flatten())
    else:
        plt.scatter(real_dist[:, 0], real_dist[:, 1])
    plt.title("real dist")

    # plot the latent representation
    plt.subplot(5, 4, 12)
    if aae_class.z_dim > 2:
        pca = PCA(n_components=2)
        pca.fit(latent_representation)
        latent_representations_current_epoch = pca.transform(latent_representation)
        plt.scatter(latent_representations_current_epoch[:, 0], latent_representations_current_epoch[:, 1])
    else:
        # plot the different classes on the latent space
        for class_label in range(n_classes):
            # get the points corresponding to the same classes
            points_for_current_class_label = latent_representation[np.where(batch_integer_labels == class_label)]
            # plot them
            plt.scatter(points_for_current_class_label[:, 0], points_for_current_class_label[:, 1],
                        label=str(class_label))
    plt.legend()
    plt.title("encoder dist")

    """
    plot the discriminator outputs
    """
    plt.subplot(5, 4, 13)
    plt.hist(discriminator_gaussian_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_gaussian_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator gaussian")
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.hist(discriminator_cat_neg.flatten(), alpha=0.5, label="neg", color="#d95f02")
    plt.hist(discriminator_cat_pos.flatten(), alpha=0.5, label="pos", color="#1b9e77")
    plt.title("discriminator categorical")
    plt.legend()

    """
    plot the learning rates over time
    """
    plt.subplot(5, 4, 16)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["autoencoder_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["autoencoder_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("autoencoder_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 17)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_g_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["discriminator_g_lr"][:n_points_to_plot], alpha=0.5,
                     c="gray")
    plt.title("discriminator_g_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 18)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["discriminator_c_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"], learning_rate_dict["discriminator_c_lr"], alpha=0.5,
                     c="gray")
    plt.title("discriminator_c_lr")
    plt.xlabel("epoch")

    plt.subplot(5, 4, 19)
    plt.plot(aae_class.learning_rates["list_of_epochs"],
             aae_class.learning_rates["generator_lr"])
    # plot the performance of the other adv autoencoders
    if include_tuning_performance:
        # get the dictionary holding the dictionaries which holds the learning rates over time
        learning_rates = Storage.get_tuning_results_learning_rates_over_time()
        for key, learning_rate_dict in learning_rates.items():
            n_points_to_plot = len([i for i in learning_rate_dict["list_of_epochs"] if i <= epoch + 1])
            plt.plot(learning_rate_dict["list_of_epochs"][:n_points_to_plot],
                     learning_rate_dict["generator_lr"][:n_points_to_plot], alpha=0.5, c="gray")
    plt.title("generator_lr")
    plt.xlabel("epoch")

    # save the figure in the results folder
    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + str(epoch)
                + "_" + str(mini_batch_i) + '.png')
    plt.close('all')

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


def create_reconstruction_grid(aae_class, real_images, reconstructed_images, epoch):
    """
    creates the reconstruction grid showing the input images and their respective reconstruction
    :param aae_class: AAE instance to access some important fields from it
    :param real_images: array of the input images
    :param reconstructed_images: array of the reconstructed images
    :param epoch: epoch of the training; used for filename and title
    :return:
    """

    nx, ny = 10, 10
    if aae_class.batch_size < 100:
        nx, ny = np.floor(np.sqrt(aae_class.batch_size)) * 2

    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    # counter for the two arrays holding the real and the reconstruction images
    image_array_counter = 0

    # iterate over the image grid
    for i, g in enumerate(gs):

        ax = plt.subplot(g)

        if i % 2 == 0:
            # plot one input image..
            real_img = real_images[image_array_counter, :]
            img = reshape_image_array(aae_class, real_img)
            if aae_class.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
        else:
            # .. and its reconstruction
            created_img = reconstructed_images[image_array_counter, :]
            img = reshape_image_array(aae_class, created_img)
            if aae_class.color_scale == "gray_scale":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            # we plotted one input image and its reconstruction, so we go on to the next images
            image_array_counter += 1

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

    plt.savefig(aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/'
                + str(epoch) + "_" + "_reconstruction_grid" + '.png')

    plt.close("all")


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    :param text: text sorted in human order ()
    :return:
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def create_gif(aae_class):
    """
    creates a gif showing the learning progress on the latent space and the class distribution on it
    :param aae_class: instance of (Un/Semi)-supervised adversarial autoencoder, holding the parameters
    :return:
    """
    result_path = aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/'
    filenames = glob.glob(result_path + "*_latent_space_class_distribution.png")
    filenames.sort(key=natural_keys)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    if len(images) > 0:
        imageio.mimwrite(result_path + 'latent_space_class_distribution.gif', images, duration=1.0)


def visualize_autoencoder_weights_and_biases(aae_class, epoch):
    """
    visualizes the autoencoder weights and biases by drawing histograms of them and plotting the weights of the input
    layer
    :param aae_class: AAE instance to access some important fields from it
    :param epoch: epoch we currently in (for the filename)
    :return:
    """

    # get all the weights
    all_weights = get_all_layer_vars(aae_class, "weights")

    # get all the bias
    all_biases = get_all_layer_vars(aae_class, "bias")

    # create the path where the weight images should be stored
    result_path_weight_img = aae_class.results_path + aae_class.result_folder_name + '/Tensorboard/' + \
                             str(epoch) + "_"

    # visualize the weights
    visualize_single_layer_with_histogram(all_weights, "encoder_dense_layer_1", "weights", result_path_weight_img,
                                          epoch)

    # visualize the biases
    visualize_single_layer_with_histogram(all_biases, "encoder_dense_layer_1", "bias", result_path_weight_img, epoch)

    # visualize the input layer weights
    input_layer_weights = all_weights["encoder_dense_layer_1"]
    visualize_input_layer_weights(input_layer_weights, result_path_weight_img, aae_class.input_dim_x,
                                  aae_class.input_dim_y, aae_class.color_scale, epoch)

    # TODO: experimental!
    if False:
        input_layer_weights = all_weights["encoder_dense_layer_1"]
        visualize_weights_single_layer_as_img_grid(input_layer_weights, aae_class.input_dim, "encoder_dense_layer_1",
                                                   "weights", result_path_weight_img, epoch)


def visualize_input_layer_weights(weights, result_path, input_dim_x, input_dim_y, color_scale, epoch):
    """
    visualizes the weights of the input layer of the encoder by plotting them on one image (gray scale) or three
    separate images for each channel (rgb scale)
    :param weights: numpy array of the weights with shape (input_dim, n_neurons_first_layer_encoder)
    :param result_path: the path for saving the image(s)
    :param input_dim_x: input dimension for x direction
    :param input_dim_y: input dimension for y direction
    :param color_scale: which color scale the data uses ("gray_scale" or "rgb_scale")
    :param epoch: current epoch (for the plot title)
    :return:
    """

    # calculate the mean weights for each input pixel
    means_per_row = np.mean(weights, axis=1)

    if color_scale == "gray_scale":
        # reshape the array for matplotlib
        means_per_row_img = means_per_row.reshape(input_dim_x, input_dim_y)

        # plot the array and save it
        plt.imshow(means_per_row_img, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights.png")
        plt.close()
    else:
        # get the number of colored pixels for each channel
        n_colored_pixels_per_channel = input_dim_x * input_dim_y

        # first n_colored_pixels_per_channel encode red
        red_pixels = means_per_row[:n_colored_pixels_per_channel].reshape(input_dim_x, input_dim_y)
        # next n_colored_pixels_per_channel encode green
        green_pixels = means_per_row[n_colored_pixels_per_channel:n_colored_pixels_per_channel * 2].reshape(input_dim_x,
                                                                                                            input_dim_y)
        # last n_colored_pixels_per_channel encode blue
        blue_pixels = means_per_row[n_colored_pixels_per_channel * 2:].reshape(input_dim_x, input_dim_y)

        # grey scale luminosity conversion formula is 0.21 R + 0.72 G + 0.07 B
        all_channels = red_pixels * 0.21 + green_pixels * 0.72 + blue_pixels * 0.07

        # plot the arrays for the different channels and save it
        plt.imshow(red_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_red_channel.png")
        plt.close()

        plt.imshow(green_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_green_channel.png")
        plt.close()

        plt.imshow(blue_pixels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_blue_channel.png")
        plt.close()

        plt.imshow(all_channels, cmap="gray")
        plt.title("Epoch: " + str(epoch))
        plt.savefig(result_path + "input_weights_all_channels.png")
        plt.close()


def visualize_single_layer_with_histogram(all_values_for_layers, layer_name, var_to_visualize, result_path, epoch):
    """
    visualizes the layer variables (weights or biases) with a histogram
    :param all_values_for_layers: numpy array holding the weights or the biases of all layers
    :param layer_name: name of the layer to visualize
    :param var_to_visualize: one of ["weights", "bias"]
    :param result_path: path for saving the images to
    :param epoch: current epoch (for the plot title)
    :return:
    """

    value_for_layer = all_values_for_layers[layer_name]

    print("visualize_single_layer_with_histogram")
    print(value_for_layer.shape)

    plt.hist(value_for_layer.flatten())
    plt.title("Epoch: " + str(epoch))
    plt.savefig(result_path + layer_name + "_" + var_to_visualize + ".png")
    plt.close()


def visualize_weights_single_layer_as_img_grid(weights, input_dim, layer_name, var_to_visualize, result_path, epoch):

    # TODO: color scale!
    #

    # we assume quadratic input images
    nx = int(np.sqrt(input_dim))

    # increase figure size
    plt.rcParams["figure.figsize"] = (30, 30)

    plt.subplot()
    gs = gridspec.GridSpec(nx, nx, hspace=0.05, wspace=0.05)

    for i, grid in enumerate(gs):

        weight = weights[i]

        # we want a quadratic image..
        # TODO: deal with hard coded stuff
        weight = weight[:961]

        weight_img = np.reshape(weight, (31, 31))

        # img = np.array(img_array).reshape(aae_class.input_dim_x, aae_class.input_dim_y)

        # print(i)

        ax = plt.subplot(gs[i])
        ax.imshow(weight_img, cmap="gray")

    plt.title("Epoch: " + str(epoch))
    plt.savefig(result_path + layer_name + "_" + var_to_visualize + "image_grid.png")

    # change figsize back to default
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

# TODO: maybe generate image grid unsupervised
# TODO: maybe generate image grid (semi-)supervised
# TODO: maybe generate image grid z_dim

