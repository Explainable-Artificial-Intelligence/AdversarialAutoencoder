import AdversarialAutoencoder
import AdversarialAutoencoderParameters
import random


def do_gridsearch(*args, **kwargs):
    """
    Performs a grid search using all possible combinations of the parameters provided. In case there are no parameters
    provided it uses all the possible parameter combinations from the hard coded parameters.
    Example calls:
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.1, 0.01, 0.001],
                    MomentumOptimizer_momentum_autoencoder=[1.0, 0.9, 0.8])
        - do_gridsearch(n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                        n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])
        - do_gridsearch(n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", "learning_rate_autoencoder")
        - do_gridsearch()
        - do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5],
                        MomentumOptimizer_momentum_autoencoder=[1.0])
    :param args: strings of the variable defined in the Parameters class to do the grid search for. In this case
    it uses the possible parameter values in the Parameters class: "variable_name"
    :param kwargs: arbitrary number of: variable_name=[variable_value1, variable_value2, variable_value3]
    :return: the best parameter combination as a dictionary
    """

    print("Doing grid search..")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_parameter_class.get_gridsearch_parameters(*args, **kwargs)

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(gridsearch_parameter_combinations), "combinations:")
    for a in gridsearch_parameter_combinations:
        print(a)
    print()
    return

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

    return sorted_list[0][0]


def do_randomsearch(n_parameter_combinations=5, *args, **kwargs):
    """
    Performs a random search using n_parameter_combinations different parameter combinations. The parameter combination
    is obtained by randomly assigning values for the parameters provided (args and kwargs).
    Example calls:
        - do_randomsearch()
        - do_randomsearch(2, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
        - do_randomsearch(10, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
        - do_randomsearch(5, "batch_size", "learning_rate_autoencoder")
        - do_randomsearch(5, learning_rate_autoencoder=random.uniform(0.2, 0.001),
                          learning_rate_discriminator=random.uniform(0.2, 0.001))
    :param n_parameter_combinations: number of parameter combinations to try
    :param args: strings of the variable defined in the Parameters class to randomize
    :param kwargs: manually assigned values for the specified variable
    :return: the best parameter combination as a dictionary
    """

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

    print("There are", len(random_param_combinations), "combinations:")
    for a in random_param_combinations:
        print(a)
    print()

    return

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

    return sorted_list[0][0]


def testing():
    """
    :return:
    """

    """
    test random search
    """
    do_randomsearch()

    do_randomsearch(2, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))

    do_randomsearch(10, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))

    do_randomsearch(5, "batch_size", "learning_rate_autoencoder")

    do_randomsearch(5, learning_rate_autoencoder=random.uniform(0.2, 0.001),
                    learning_rate_discriminator=random.uniform(0.2, 0.001))

    """
    test grid search
    """
    do_gridsearch(n_neurons_of_hidden_layer_x_autoencoder=[[500, 250, 125], [1000, 750, 25]],
                  n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])

    do_gridsearch(n_neurons_of_hidden_layer_x_discriminator=[[500, 250, 125], [1000, 750, 25]])

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", "learning_rate_autoencoder")

    do_gridsearch()

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5],
                  MomentumOptimizer_momentum_autoencoder=[1.0])


if __name__ == '__main__':
    testing()
