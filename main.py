import AdversarialAutoencoder
import AdversarialAutoencoderParameters


def do_gridsearch(*args, **kwargs):

    print("Doing grid search..")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_parameter_class.get_gridsearch_parameters(*args, **kwargs)

    # n_neurons_of_hidden_layer_x_autoencoder = [1000, 500, 250]
    # n_neurons_of_hidden_layer_x_discriminator = [500, 250, 125]

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(gridsearch_parameter_combinations), "combinations:")
    for a in gridsearch_parameter_combinations:
        print(a)

    # iterate over each parameter combination
    for gridsearch_parameter_combination in gridsearch_parameter_combinations:
        print("Training .. ", gridsearch_parameter_combination)

        # create the AAE and train it with the current parameters
        adv_autoencoder = AdversarialAutoencoder. \
            AdversarialAutoencoder(gridsearch_parameter_combination)
        adv_autoencoder.train(True)

        # get the performance
        performance = adv_autoencoder.get_performance()
        print(performance)

        # store the param_comb and the performance in the list
        performance_for_parameter_combination.append((gridsearch_parameter_combination, performance))

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x[1]["summed_loss_final"])

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])


def do_randomsearch(n_parameter_combinations=5, *args, **kwargs):

    print("Doing random search..")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # get some random parameter combinations
    random_param_combinations = [aae_parameter_class.get_randomized_parameters(*args, **kwargs)
                                 for i in range(1, n_parameter_combinations)]

    # add the default combination to the list
    random_param_combinations.append(aae_parameter_class.get_default_parameters())

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    # iterate over each parameter combination
    for random_param_combination in random_param_combinations:
        print(random_param_combination)

        # create the AAE and train it with the current parameters
        adv_autoencoder = AdversarialAutoencoder. \
            AdversarialAutoencoder(random_param_combination)
        adv_autoencoder.train(True)

        # get the performance
        performance = adv_autoencoder.get_performance()
        print(performance)

        # store the param_comb and the performance in the list
        performance_for_parameter_combination.append((random_param_combination, performance))

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x[1]["summed_loss_final"])

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])


def testing():

    do_randomsearch(2, "batch_size", learning_rate_autoencoder=0.1)

    do_gridsearch(learning_rate_autoencoder=[0.01])




if __name__ == '__main__':
    testing()
