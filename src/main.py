from util.TuningFunctions import do_randomsearch, do_gridsearch


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

    do_randomsearch(1, selected_autoencoder="Supervised", selected_dataset="cifar10", n_epochs=6, verbose=True,
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
