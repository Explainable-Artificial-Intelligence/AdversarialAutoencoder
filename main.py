import AdversarialAutoencoder
import AdversarialAutoencoderParameters


def do_randomsearch(n_parameter_combinations):

    aae_parameter_class = AdversarialAutoencoderParameters.AdversarialAutoencoderParameters()
    random_param_combinations = [aae_parameter_class.get_randomized_parameters()
                                 for i in range(0, n_parameter_combinations)]

    random_param_combinations.append(aae_parameter_class.get_default_parameters())
    performance_for_parameter_combination = []

    for random_param_combination in random_param_combinations:
        print(random_param_combination)

        # input data parameters
        random_param_combination["input_dim"] = 784
        random_param_combination["z_dim"] = 2

        # path to the results
        random_param_combination["results_path"] = './Results'

        adv_autoencoder = AdversarialAutoencoder. \
            AdversarialAutoencoder(random_param_combination)

        adv_autoencoder.train(True)

        performance = adv_autoencoder.get_performance()
        print(performance)

        performance_for_parameter_combination.append((random_param_combination, performance))

    # sort combinations by their performance
    sorted_list = sorted(performance_for_parameter_combination, key=lambda x: x[1]["summed_loss_final"])

    print(sorted_list)
    print("best param combination:", sorted_list[0][0])
    print("best performance:", sorted_list[0][1])


def testing():

    do_randomsearch(2)


if __name__ == '__main__':
    testing()
