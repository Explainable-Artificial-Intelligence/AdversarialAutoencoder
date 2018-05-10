import threading
import connexion
from swagger_server.utils.Storage import Storage


def control_training(train_status):
    """
    starts and stops the training
    :param train_status: one of ["start", "stop"]; whether to start or stop the training
    :return:
    """
    if connexion.request.is_json:
        if train_status == "start":
            # get the adv. autoencoder
            aae = Storage.get_aae()

            # set the train status
            aae.set_train_status("start")

            try:
                # define background thread:
                aae_thread = threading.Thread(target=aae.train, args=(True,))
            except AttributeError:
                return "No autoencoder found", 404

            # the adv autoencoder is currently training, so we need to stop it first
            if Storage.get_aae_thread():
                # stop training
                aae.set_train_status("stop")

            # set the new training thread
            Storage.set_aae_thread(aae_thread)

            # start training:
            aae_thread.start()

            return "Training started", 200

        if train_status == "stop":
            # get aae
            aae = Storage.get_aae()

            # stop training
            aae.set_train_status("stop")

            return "Training stopped", 200


def get_performance_over_time():
    """
    returns the performance over time (losses (and accuracy for the semi-supervised aae)) for the current autoencoder
    :return:
    """

    # get the adversarial autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: no autoencoder found", 404

    # get the performance over time
    performance_over_time = aae.get_performance_over_time()

    # since swagger doesn't allow different return values for the same function, we return all of them
    discriminator_losses = []                   # only (un)-supervised
    discriminator_gaussian_losses = []          # only semi-supervised
    discriminator_categorical_losses = []       # only semi-supervised
    supervised_encoder_loss = []                # only semi-supervised
    accuracy = []                               # only semi-supervised
    accuracy_epochs = []                        # only semi-supervised

    # distinguish between semi-supervised or (un-)supervised autoencoder
    if Storage.get_selected_autoencoder() == "SemiSupervised":

        autoencoder_losses = performance_over_time["autoencoder_losses"]
        autoencoder_losses = [float(number) for number in autoencoder_losses]

        discriminator_gaussian_losses = performance_over_time["discriminator_gaussian_losses"]
        discriminator_gaussian_losses = [float(number) for number in discriminator_gaussian_losses]

        discriminator_categorical_losses = performance_over_time["discriminator_categorical_losses"]
        discriminator_categorical_losses = [float(number) for number in discriminator_categorical_losses]

        generator_losses = performance_over_time["generator_losses"]
        generator_losses = [float(number) for number in generator_losses]

        supervised_encoder_loss = performance_over_time["supervised_encoder_loss"]
        supervised_encoder_loss = [float(number) for number in supervised_encoder_loss]

        accuracy = performance_over_time["accuracy"]
        accuracy = [float(number) for number in accuracy]

        accuracy_epochs = performance_over_time["accuracy_epochs"]
        accuracy_epochs = [float(number) for number in accuracy_epochs]

        list_of_epochs = performance_over_time["list_of_epochs"]
        list_of_epochs = [float(number) for number in list_of_epochs]

    # we have an unsupervised or a supervised autoencoder
    else:

        autoencoder_losses = performance_over_time["autoencoder_losses"]
        autoencoder_losses = [float(number) for number in autoencoder_losses]

        discriminator_losses = performance_over_time["discriminator_losses"]
        discriminator_losses = [float(number) for number in discriminator_losses]

        generator_losses = performance_over_time["generator_losses"]
        generator_losses = [float(number) for number in generator_losses]

        list_of_epochs = performance_over_time["list_of_epochs"]
        list_of_epochs = [float(number) for number in list_of_epochs]

    # since swagger doesn't allow multiple return values, we have to pack them in a dictionary and return it
    performance_dict = {"autoencoder_losses:": autoencoder_losses, "discriminator_losses:": discriminator_losses,
                        "generator_losses:": generator_losses, "list_of_epochs:": list_of_epochs,
                        "discriminator_gaussian_losses": discriminator_gaussian_losses,
                        "discriminator_categorical_losses": discriminator_categorical_losses,
                        "supervised_encoder_loss": supervised_encoder_loss, "accuracy": accuracy,
                        "accuracy_epochs": accuracy_epochs}

    return performance_dict, 200


def get_learning_rates():
    """
    returns the learning rates over time for the current autoencoder
    :return:
    """

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: no autoencoder found", 404

    # get the learning rates
    learning_rates = aae.get_learning_rates()

    # since swagger doesn't allow different return values for the same function, we return all of them
    discriminator_lr = []       # only (un)-supervised
    discriminator_g_lr = []     # only semi-supervised
    discriminator_c_lr = []     # only semi-supervised
    supervised_encoder_lr = []  # only semi-supervised

    # distinguish between semi-supervised or (un-)supervised autoencoder
    if Storage.get_selected_autoencoder() == "SemiSupervised":
        autoencoder_lr = learning_rates["autoencoder_lr"]
        autoencoder_lr = [float(number) for number in autoencoder_lr]

        discriminator_g_lr = learning_rates["discriminator_g_lr"]
        discriminator_g_lr = [float(number) for number in discriminator_g_lr]

        discriminator_c_lr = learning_rates["discriminator_c_lr"]
        discriminator_c_lr = [float(number) for number in discriminator_c_lr]

        generator_lr = learning_rates["generator_lr"]
        generator_lr = [float(number) for number in generator_lr]

        supervised_encoder_lr = learning_rates["supervised_encoder_lr"]
        supervised_encoder_lr = [float(number) for number in supervised_encoder_lr]

        list_of_epochs = learning_rates["list_of_epochs"]
        list_of_epochs = [float(number) for number in list_of_epochs]

    # we have an unsupervised or a supervised autoencoder
    else:

        autoencoder_lr = learning_rates["autoencoder_lr"]
        autoencoder_lr = [float(number) for number in autoencoder_lr]

        discriminator_lr = learning_rates["discriminator_lr"]
        discriminator_lr = [float(number) for number in discriminator_lr]

        generator_lr = learning_rates["generator_lr"]
        generator_lr = [float(number) for number in generator_lr]

        list_of_epochs = learning_rates["list_of_epochs"]
        list_of_epochs = [float(number) for number in list_of_epochs]

    # since swagger doesn't allow multiple return values, we have to pack them in a dictionary and return it
    lr_dict = {"autoencoder_lr:": autoencoder_lr, "discriminator_lr:": discriminator_lr,
               "generator_lr:": generator_lr, "list_of_epochs:": list_of_epochs,
               "discriminator_g_lr": discriminator_g_lr, "discriminator_c_lr": discriminator_c_lr,
               "supervised_encoder_lr": supervised_encoder_lr}

    return lr_dict, 200


def get_minibatch_summary_vars():

    # get the autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: no autoencoder found", 404

    # get the vars for the minibatch summary
    minibatch_summary_vars = aae.get_minibatch_summary_vars()

    # since swagger doesn't allow different return values for the same function, we return all of them
    discriminator_neg = []       # only (un)-supervised
    discriminator_pos = []       # only (un)-supervised
    batch_x = []       # only (un)-supervised
    decoder_output = []       # only (un)-supervised
    batch_labels = []       # only (un)-supervised

    batch_X_unlabeled = []     # only semi-supervised
    reconstructed_image = []     # only semi-supervised
    real_cat_dist = []     # only semi-supervised
    encoder_cat_dist = []     # only semi-supervised
    batch_X_unlabeled_labels = []     # only semi-supervised
    discriminator_gaussian_neg = []     # only semi-supervised
    discriminator_gaussian_pos = []     # only semi-supervised
    discriminator_cat_neg = []     # only semi-supervised
    discriminator_cat_pos = []     # only semi-supervised

    # distinguish between semi-supervised or (un-)supervised autoencoder
    if Storage.get_selected_autoencoder() == "SemiSupervised":

        real_dist = minibatch_summary_vars["real_dist"]  # (batch_size, z_dim) array of floats
        real_dist = real_dist.astype("float64").tolist()

        latent_representation = minibatch_summary_vars["latent_representation"]  # (batch_size, z_dim) array of floats
        latent_representation = latent_representation.astype("float64").tolist()

        batch_X_unlabeled = minibatch_summary_vars["batch_X_unlabeled"]  # (batch_size, z_dim) array of floats
        batch_X_unlabeled = batch_X_unlabeled.astype("float64").tolist()

        reconstructed_image = minibatch_summary_vars["latent_representation"]  # (batch_size, z_dim) array of floats
        reconstructed_image = reconstructed_image.astype("float64").tolist()

        real_cat_dist = minibatch_summary_vars["real_cat_dist"]  # (batch_size, z_dim) array of floats
        real_cat_dist = real_cat_dist.astype("float64").tolist()

        encoder_cat_dist = minibatch_summary_vars["encoder_cat_dist"]  # (batch_size, z_dim) array of floats
        encoder_cat_dist = encoder_cat_dist.astype("float64").tolist()

        batch_X_unlabeled_labels = minibatch_summary_vars["batch_X_unlabeled_labels"]  # (batch_size, z_dim) array of floats
        batch_X_unlabeled_labels = batch_X_unlabeled_labels.astype("float64").tolist()

        discriminator_gaussian_neg = minibatch_summary_vars["discriminator_gaussian_neg"]  # (batch_size) array of floats
        discriminator_gaussian_neg = discriminator_gaussian_neg.astype("float64").tolist()

        discriminator_gaussian_pos = minibatch_summary_vars["discriminator_gaussian_pos"]  # (batch_size) array of floats
        discriminator_gaussian_pos = discriminator_gaussian_pos.astype("float64").tolist()

        discriminator_cat_neg = minibatch_summary_vars["discriminator_cat_neg"]  # (batch_size) array of floats
        discriminator_cat_neg = discriminator_cat_neg.astype("float64").tolist()

        discriminator_cat_pos = minibatch_summary_vars["discriminator_cat_pos"]  # (batch_size, z_dim) array of floats
        discriminator_cat_pos = discriminator_cat_pos.astype("float64").tolist()

        epoch = minibatch_summary_vars["epoch"]  # single integer
        b = minibatch_summary_vars["b"]  # single integer

    # we have an unsupervised or a supervised autoencoder
    else:
        real_dist = minibatch_summary_vars["real_dist"]  # (batch_size, z_dim) array of floats
        real_dist = real_dist.astype("float64").tolist()

        latent_representation = minibatch_summary_vars["latent_representation"]  # (batch_size, z_dim) array of floats
        latent_representation = latent_representation.astype("float64").tolist()

        discriminator_neg = minibatch_summary_vars["discriminator_neg"]  # (batch_size) array of floats
        discriminator_neg = discriminator_neg.astype("float64").tolist()

        discriminator_pos = minibatch_summary_vars["discriminator_pos"]  # (batch_size, z_dim) array of floats
        discriminator_pos = discriminator_pos.astype("float64").tolist()

        batch_x = minibatch_summary_vars["batch_x"]  # (batch_size, input_dim_x*input_dim_x*color_scale) array of floats
        batch_x = batch_x.astype("float64").tolist()
        decoder_output = minibatch_summary_vars["x_reconstructed"]   # (batch_size, input_dim_x*input_dim_x*color_scale)
        decoder_output = decoder_output.astype("float64").tolist()  # array of floats

        batch_labels = minibatch_summary_vars["batch_labels"]  # (batch_size, n_classes) array of ints
        batch_labels = batch_labels.astype("float64").tolist()

        epoch = minibatch_summary_vars["epoch"]  # single integer
        b = minibatch_summary_vars["b"]  # single integer

    minibatch_summary_vars_dict = {"real_dist": real_dist, "latent_representation": latent_representation,
                                   "discriminator_neg": discriminator_neg, "discriminator_pos": discriminator_pos,
                                   "batch_x": batch_x, "x_reconstructed": decoder_output, "epoch": epoch, "b": b,
                                   "batch_labels": batch_labels, "batch_X_unlabeled": batch_X_unlabeled,
                                   "reconstructed_image": reconstructed_image, "real_cat_dist": real_cat_dist,
                                   "encoder_cat_dist": encoder_cat_dist,
                                   "batch_X_unlabeled_labels": batch_X_unlabeled_labels,
                                   "discriminator_gaussian_neg": discriminator_gaussian_neg,
                                   "discriminator_gaussian_pos": discriminator_gaussian_pos,
                                   "discriminator_cat_neg": discriminator_cat_neg,
                                   "discriminator_cat_pos": discriminator_cat_pos}



    # TODO
    return minibatch_summary_vars_dict, 200


def reset_tensorflow_graph():
    """
    resets the tensorflow graph to enable training another autoencoder
    :return:
    """

    # get the adversarial autoencoder
    aae = Storage.get_aae()

    # check if we have an autoencoder
    if not aae:
        return "Error: no autoencoder found", 404

    aae.reset_graph()

    return "Graph successfully reset", 200
