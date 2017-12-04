import AdversarialAutoencoder


def testing():
    # parameters
    input_dim = 784
    n_neurons_of_hidden_layer_x = [1000, 1000]
    z_dim = 2
    batch_size = 100
    n_epochs = 5
    learning_rate = 0.001
    beta1 = 0.9
    results_path = './Results'

    # loss functions
    loss_functions = ["absolute_difference",
                      "hinge_loss",
                      "mean_squared_error",
                      "sigmoid_cross_entropy"]

    loss_function = "sigmoid_cross_entropy"

    print("####################################################################################################")
    print("loss function:" + loss_function)

    adv_autoencoder = AdversarialAutoencoder. \
        AdversarialAutoencoder(input_dim=input_dim, n_neurons_of_hidden_layer_x=n_neurons_of_hidden_layer_x,
                               z_dim=z_dim, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
                               beta1=beta1, loss_function=loss_function, results_path=results_path)

    adv_autoencoder.train(True)


if __name__ == '__main__':
    testing()
