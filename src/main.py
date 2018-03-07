import json

import util.AdversarialAutoencoderParameters as aae_params

from autoencoders.SemiSupervisedAdversarialAutoencoder import SemiSupervisedAdversarialAutoencoder
from autoencoders.SupervisedAdversarialAutoencoder import SupervisedAdversarialAutoencoder
from autoencoders.UnsupervisedAdversarialAutoencoder import AdversarialAutoencoder


def init_aae_with_params_file(params_filename, used_aae):
    """
    returns a adversarial autoencoder initiated with the provided parameter file
    :param params_filename: path to the params.txt file
    :param used_aae: ["Unsupervised", "Supervised", "SemiSupervised"]
    :return:
    """

    # get the used parameters from the params.txt file
    used_params = json.load(open(params_filename))

    # create the AAE and train it with the used parameters
    adv_autoencoder = None
    if used_aae == "Unsupervised":
        adv_autoencoder = AdversarialAutoencoder(used_params)
    elif used_aae == "Supervised":
        adv_autoencoder = SupervisedAdversarialAutoencoder(used_params)
    elif used_aae == "SemiSupervised":
        adv_autoencoder = SemiSupervisedAdversarialAutoencoder(used_params)

    return adv_autoencoder


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
    aae_parameter_class = aae_params.AdversarialAutoencoderParameters()

    # iterate over the parameter combinations
    gridsearch_parameter_combinations = \
        aae_parameter_class.get_gridsearch_parameters(*args, selected_autoencoder=selected_autoencoder,
                                                      selected_dataset=selected_dataset, **kwargs)

    # stores the performance for the parameter combination
    performance_for_parameter_combination = []

    print("There are", len(gridsearch_parameter_combinations), "combinations:")

    # for a in gridsearch_parameter_combinations:
    #     print(a)
    # print()

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
        performance = adv_autoencoder.get_final_performance()
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
        with open('gridsearch_log.txt', 'a') as log:
            log.write("performance: {}\n".format(comb[1]))
            log.write("folder name: {}\n".format(comb[2]))

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])
    print("folder name:", sorted_list[0][2])

    with open('gridsearch_log.txt', 'a') as log:
        log.write("best param combination: {}\n".format(sorted_list[0][0]))
        log.write("best performance: {}\n".format(sorted_list[0][1]))
        log.write("folder name: {}\n".format(sorted_list[0][2]))

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
    aae_parameter_class = aae_params.AdversarialAutoencoderParameters()

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
        adv_autoencoder.train(False)

        # get the performance
        performance = adv_autoencoder.get_final_performance()
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


def create_network_topology(n_layers, init_n_neurons, n_neurons_decay_factor, n_decaying_layers):
    """
    creates a network topology based on the provided parameters
    :param n_layers: number of layers the network should have
    :param init_n_neurons: number of neurons the first layer should have
    :param n_neurons_decay_factor: by what factor the number of neurons in the suceeding layers should be reduced
    e.g. with a factor of 2: [3000, 3000, 3000] -> [3000, 1500, 750]
    :param n_decaying_layers: number of layers where the number of neurons should be reduced by the
    n_neurons_decay_factor
    :return:
    """

    random_network_topology = [init_n_neurons]*n_layers
    for decaying_layer in range(n_decaying_layers, 0, -1):
        random_network_topology[n_layers-decaying_layer] = \
             int(random_network_topology[n_layers-decaying_layer-1] / n_neurons_decay_factor)

    return random_network_topology


def create_random_network_topologies(max_layers, init_n_neurons, n_neurons_decay_factors):

    random_network_topologies = []

    for n_layers in range(1, max_layers+1):
        # maximum number of layers with a reduced number of nerons compared to the preceding layers
        for n_decaying_layers in range(n_layers):

            # we only have to iterate over the n_neurons_decay_factors if we have at least one decaying layer
            if n_decaying_layers > 0:
                for n_neurons_decay_factor in n_neurons_decay_factors:
                    random_network_topologies.append(
                            create_network_topology(n_layers, init_n_neurons, n_neurons_decay_factor, n_decaying_layers))
            # otherwise we can pick an arbitrary value for the n_neurons_decay_factor
            else:
                random_network_topologies.append(
                    create_network_topology(n_layers, init_n_neurons, 1, n_decaying_layers))

    return random_network_topologies


def testing():
    """
    :return:
    """

    selected_datasets = ["MNIST", "SVHN", "cifar10", "custom"]
    selected_autoencoders = ["Unsupervised", "Supervised", "SemiSupervised"]
    decaying_learning_rate_names = ["exponential_decay", "inverse_time_decay", "natural_exp_decay",
                                    "piecewise_constant", "polynomial_decay", "static"]
    activation_functions = ["relu", "relu6", "crelu", "elu", "softplus", "softsign", "sigmoid", "tanh", "linear"]


    # aae = init_aae_with_params_file("C:\\Users\\Telcontar\\Desktop\\Good_Results\\2018-02-13_10_48_53_SVHN\\log\\params.txt", "Supervised")
    # aae.train(True)

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="cifar10", n_epochs=6, verbose=True,
                    z_dim=2, batch_size=100, save_final_model=True,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    activation_function_encoder=['relu']*4,
                    activation_function_decoder='relu',
                    activation_function_discriminator='relu',
                    decaying_learning_rate_name_autoencoder="static",
                    decaying_learning_rate_name_discriminator="static",
                    decaying_learning_rate_name_generator="static",
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    # n_neurons_of_hidden_layer_x_autoencoder=[1000, 1000],
                    # n_neurons_of_hidden_layer_x_discriminator=[1000, 1000],
                    # n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    # n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    # bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    # bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

    return

    # try overfitting

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="cifar10", n_epochs=10000, verbose=True,
                    z_dim=32,
                    learning_rate_autoencoder=0.0001,
                    learning_rate_discriminator=0.0001,
                    learning_rate_generator=0.0001,
                    decaying_learning_rate_name_autoencoder=None,
                    decaying_learning_rate_name_discriminator=None,
                    decaying_learning_rate_name_generator=None,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 3000, 3000],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 3000, 3000],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0]
                    )

    return

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="SVHN", n_epochs=100, verbose=True,
                    z_dim=2,
                    learning_rate_autoencoder=0.01,
                    learning_rate_discriminator=0.01,
                    learning_rate_generator=0.01,
                    AdamOptimizer_beta1_autoencoder=0.5,
                    AdamOptimizer_beta1_discriminator=0.5,
                    AdamOptimizer_beta1_generator=0.5,
                    n_neurons_of_hidden_layer_x_autoencoder=[3000, 2000, 1000, 500, 250, 125],
                    n_neurons_of_hidden_layer_x_discriminator=[3000, 2000, 1000, 500, 250, 125],
                    bias_init_value_of_hidden_layer_x_autoencoder=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    bias_init_value_of_hidden_layer_x_discriminator=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

    return

    n_neurons_of_hidden_layer_x_autoencoder = \
        create_random_network_topologies(max_layers=3, init_n_neurons=3000,
                                         n_neurons_decay_factors=[1, 1.5, 2, 3])

    n_neurons_of_hidden_layer_x_discriminator = \
        create_random_network_topologies(max_layers=3, init_n_neurons=3000,
                                         n_neurons_decay_factors=[1, 1.5, 2, 3])

    bias_init_value_of_hidden_layer_x_autoencoder = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_autoencoder]
    bias_init_value_of_hidden_layer_x_discriminator = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_discriminator]

    do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=[100], verbose=False,
                  z_dim=[2],
                  learning_rate_autoencoder=[0.001],
                  learning_rate_discriminator=[0.01],
                  learning_rate_generator=[0.01],
                  AdamOptimizer_beta1_autoencoder=[0.5],
                  AdamOptimizer_beta1_discriminator=[0.5],
                  AdamOptimizer_beta1_generator=[0.5],
                  n_neurons_of_hidden_layer_x_autoencoder=n_neurons_of_hidden_layer_x_autoencoder,
                  n_neurons_of_hidden_layer_x_discriminator=n_neurons_of_hidden_layer_x_discriminator,
                  bias_init_value_of_hidden_layer_x_autoencoder=[0.0],
                  bias_init_value_of_hidden_layer_x_discriminator=[0.0])


    return

    n_neurons_of_hidden_layer_x_autoencoder = [[3000, 2000, 1000, 500, 250, 125], [3000, 3000]]
    n_neurons_of_hidden_layer_x_discriminator = [[3000, 2000, 1000, 500, 250, 125], [3000, 3000]]
    bias_init_value_of_hidden_layer_x_autoencoder = [[0.0]*(len(i)+1) for i in n_neurons_of_hidden_layer_x_autoencoder]
    bias_init_value_of_hidden_layer_x_discriminator = [[0.0]*(len(i)+1) for i in
                                                       n_neurons_of_hidden_layer_x_discriminator]

    print(n_neurons_of_hidden_layer_x_autoencoder)
    print(n_neurons_of_hidden_layer_x_discriminator)
    print(bias_init_value_of_hidden_layer_x_autoencoder)
    print(bias_init_value_of_hidden_layer_x_discriminator)

    return

    do_gridsearch(selected_autoencoder="Unsupervised", selected_dataset="SVHN", n_epochs=[1,2], verbose=True,
                  n_neurons_of_hidden_layer_x_autoencoder=[[3000, 2000, 1000, 500, 250, 125]],
                  n_neurons_of_hidden_layer_x_discriminator=[[3000, 2000, 1000, 500, 250, 125]],
                  bias_init_value_of_hidden_layer_x_autoencoder=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                  bias_init_value_of_hidden_layer_x_discriminator=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    return

    """
    test random search
    """

    do_randomsearch(2, selected_autoencoder="Supervised", selected_dataset="MNIST", n_epochs=100, z_dim=10,
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
