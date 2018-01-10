import AdversarialAutoencoder
import AdversarialAutoencoderParameters


def do_gridsearch():

    # TODO: implement **kwargs

    print("grid search:")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_parameter_class.get_gridsearch_parameters("RMSPropOptimizer_momentum_autoencoder",
                                                      "RMSPropOptimizer_decay_autoencoder")

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    # iterate over each parameter combination
    for gridsearch_parameter_combination in gridsearch_parameter_combinations:
        print(gridsearch_parameter_combination)

        # input data parameters
        gridsearch_parameter_combination["input_dim"] = 784
        gridsearch_parameter_combination["z_dim"] = 2

        # path to the results
        gridsearch_parameter_combination["results_path"] = './Results'

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


def do_randomsearch(n_parameter_combinations):

    print("random search")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # get some random parameter combinations
    random_param_combinations = [aae_parameter_class.get_randomized_parameters()
                                 for i in range(1, n_parameter_combinations)]

    # add the default combination to the list
    random_param_combinations.append(aae_parameter_class.get_default_parameters())

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    # iterate over each parameter combination
    for random_param_combination in random_param_combinations:
        print(random_param_combination)

        # input data parameters
        random_param_combination["input_dim"] = 784
        random_param_combination["z_dim"] = 2

        # path to the results
        random_param_combination["results_path"] = './Results'

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

    # do_randomsearch(2)

    do_gridsearch()


if __name__ == '__main__':
    testing()
