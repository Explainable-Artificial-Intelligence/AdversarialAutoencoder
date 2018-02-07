import AdversarialAutoencoderParameters

from SemiSupervisedAdversarialAutoencoder import SemiSupervisedAdversarialAutoencoder
from SupervisedAdversarialAutoencoder import SupervisedAdversarialAutoencoder
from UnsupervisedAdversarialAutoencoder import AdversarialAutoencoder


def do_gridsearch(*args, selected_autoencoder="Unsupervised", selected_dataset="MNIST", **kwargs):
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
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param kwargs: arbitrary number of: variable_name=[variable_value1, variable_value2, variable_value3]
    :return: the best parameter combination as a dictionary
    """

    print("Doing grid search..")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_parameter_class.get_gridsearch_parameters(*args, selected_autoencoder=selected_autoencoder,
                                                      selected_dataset=selected_dataset, **kwargs)

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(gridsearch_parameter_combinations), "combinations:")
    for a in gridsearch_parameter_combinations:
        print(a)
    print()

    # iterate over each parameter combination
    for gridsearch_parameter_combination in gridsearch_parameter_combinations:
        print("Training .. ", gridsearch_parameter_combination)

        # create the AAE and train it with the current parameters
        if selected_autoencoder == "Unsupervised":
            adv_autoencoder = AdversarialAutoencoder(gridsearch_parameter_combination)
        elif selected_autoencoder == "Supervised":
            adv_autoencoder = SupervisedAdversarialAutoencoder(gridsearch_parameter_combination)
        elif selected_autoencoder == "SemiSupervised":
            adv_autoencoder = SemiSupervisedAdversarialAutoencoder(gridsearch_parameter_combination)
        adv_autoencoder.train(True)

        # get the performance
        performance = adv_autoencoder.get_performance()
        print(performance)
        folder_name = adv_autoencoder.get_result_folder_name()

        # store the param_comb and the performance in the list
        performance_for_parameter_combination.append((gridsearch_parameter_combination, performance, folder_name))

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x[1]["summed_loss_final"])

    print("#" * 20)

    for comb in sorted_list:
        print("performance:", comb[1])
        print("folder name:", comb[2])
        print()

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])
    print("folder name:", sorted_list[0][2])

    return sorted_list[0][0]


def do_randomsearch(n_parameter_combinations=5, *args, selected_autoencoder="Unsupervised", selected_dataset="MNIST",
                    **kwargs):
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
    :param selected_dataset: ["MNIST", "SVHN", "cifar10", "custom"]
    :param selected_autoencoder: ["Unsupervised", "Supervised", "SemiSupervised"]
    :param args: strings of the variable defined in the Parameters class to randomize
    :param kwargs: manually assigned values for the specified variable
    :return: the best parameter combination as a dictionary
    """

    print("Doing random search..")

    # get the parameter class for the AAE
    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # get some random parameter combinations
    random_param_combinations = \
        [aae_parameter_class.get_randomized_parameters(*args, selected_autoencoder=selected_autoencoder,
                                                       selected_dataset=selected_dataset, **kwargs)
         for i in range(1, n_parameter_combinations)]

    # add the default parameter combination to the list based on the selected dataset
    # random_param_combinations.append(aae_parameter_class.get_default_parameters(selected_autoencoder, selected_dataset))

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(random_param_combinations), "combinations:")
    for a in random_param_combinations:
        print(a)
    print()

    # iterate over each parameter combination
    for random_param_combination in random_param_combinations:
        print(random_param_combination)

        # create the AAE and train it with the current parameters
        if selected_autoencoder == "Unsupervised":
            adv_autoencoder = AdversarialAutoencoder(random_param_combination)
        elif selected_autoencoder == "Supervised":
            adv_autoencoder = SupervisedAdversarialAutoencoder(random_param_combination)
        elif selected_autoencoder == "SemiSupervised":
            adv_autoencoder = SemiSupervisedAdversarialAutoencoder(random_param_combination)

        adv_autoencoder.train(True)
        # adv_autoencoder.train(False)

        # get the performance
        performance = adv_autoencoder.get_performance()
        print(performance)
        folder_name = adv_autoencoder.get_result_folder_name()

        # store the param_comb and the performance in the list
        performance_for_parameter_combination.append((random_param_combination, performance, folder_name))

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x[1]["summed_loss_final"])

    print("#"*20)

    for comb in sorted_list:
        print("performance:", comb[1])
        print("folder name:", comb[2])
        print()

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])
    print("folder name:", sorted_list[0][2])

    return sorted_list[0][0]


def testing():
    """
    :return:
    """

    # do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=[1, 2], verbose=True)

    """
    test random search
    """

    selected_datasets = ["MNIST", "SVHN", "cifar10", "custom"]
    selected_autoencoders = ["Unsupervised", "Supervised", "SemiSupervised"]

    do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=1, z_dim=10,
                    verbose=True,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return

    # do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=10, z_dim=2,
    #                 verbose=True, learning_rate_autoencoder=0.01, learning_rate_discriminator=0.01,
    #                 learning_rate_generator=0.01)
    #
    # do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=10, z_dim=15,
    #                 verbose=True)

    do_randomsearch(2, selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=2000, z_dim=2,
                    verbose=True, batch_size=100,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[1000, 1000],
                    n_neurons_of_hidden_layer_x_discriminator=[1000, 1000],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0])

    # do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="MNIST", n_epochs=100, z_dim=15,
    #                 verbose=True)

    # do_randomsearch(2, selected_autoencoder="SemiSupervised", selected_dataset="MNIST", n_epochs=100, z_dim=15,
    #                 verbose=True, n_neurons_of_hidden_layer_x_autoencoder=[1536, 768, 384],
    #                 n_neurons_of_hidden_layer_x_discriminator=[1536, 768, 384],
    #                 bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0],
    #                 bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0])

    # do_randomsearch(1, selected_autoencoder="Supervised", selected_dataset="MNIST", z_dim=2)

    # do_randomsearch(10, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="Supervised", selected_dataset="SVHN",
    #                 z_dim=15, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=10, verbose=False)

    # do_randomsearch(10, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="Supervised", selected_dataset="SVHN",
    #                 z_dim=100, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=100, verbose=False)


    # do_randomsearch(100, "batch_size", "z_dim", "learning_rate_autoencoder", "learning_rate_discriminator",
    #                 "learning_rate_generator", selected_autoencoder="SemiSupervised", selected_dataset="SVHN",
    #                 z_dim=15, AdamOptimizer_beta1_autoencoder=0.5, AdamOptimizer_beta1_discriminator=0.5,
    #                 AdamOptimizer_beta1_generator=0.5, n_epochs=100, verbose=False)


    return

    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()

    # print(aae_parameter_class.draw_from_distribution(distribution_name="normal", loc=1.0, scale=2.0, n_samples=1))
    # print(aae_parameter_class.draw_from_distribution(distribution_name="uniform", low=50, high=500))
    # print(np.random.normal(loc=1.0, scale=2.0))
    # aae_parameter_class.get_randomized_parameters(batch_size={"distribution_name": "uniform", "low":50, "high":500})

    do_randomsearch(7)

    do_randomsearch(8, batch_size={"distribution_name": "uniform", "low": 50, "high": 500, "return_type": "int"})

    do_randomsearch(9, "AdamOptimizer_beta1_discriminator",
                    batch_size={"distribution_name": "normal", "loc": 150.0, "scale": 50.0, "return_type": "int",
                                "is_greater_than_zero": True},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"},
                    RMSPropOptimizer_centered_autoencoder=False)

    do_randomsearch(10, batch_size={"distribution_name": "normal", "loc": 150.0, "scale": 50.0, "return_type": "int",
                                    "is_smaller_than_zero": True},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"})

    do_randomsearch(11, batch_size={"distribution_name": "uniform", "low": 50, "high": 500, "return_type": "int"},
                    learning_rate_autoencoder={"distribution_name": "uniform", "low": 50, "high": 500,
                                               "return_type": "int"},
                    RMSPropOptimizer_centered_autoencoder=False, optimizer_autoencoder="ProximalAdagradOptimizer",
                    n_neurons_of_hidden_layer_x_autoencoder=[2587, 237, 29357])

    # do_randomsearch(100)
    # do_randomsearch(2, "batch_size", learning_rate_autoencoder=[random.uniform(0.2, 0.001)*9])
    # do_randomsearch(10, "batch_size", learning_rate_autoencoder=random.uniform(0.2, 0.001))
    # do_randomsearch(5, "batch_size", "learning_rate_autoencoder")
    # do_randomsearch(10, batch_size=random.uniform(0.2, 0.001))
    # do_randomsearch(5, learning_rate_autoencoder=random.uniform(0.2, 0.001),
    #                 learning_rate_discriminator=random.uniform(0.2, 0.001))

    return

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

    do_gridsearch("n_neurons_of_hidden_layer_x_autoencoder", learning_rate_autoencoder=[0.5, 0.1, 0.01, 0.001],
                  MomentumOptimizer_momentum_autoencoder=[1.0])


if __name__ == '__main__':
    testing()
