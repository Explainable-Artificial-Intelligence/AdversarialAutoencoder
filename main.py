import AdversarialAutoencoder


def testing():
    # parameters
    input_dim = 784
    z_dim = 2
    batch_size = 100
    n_neurons_of_hidden_layer_x_autoencoder = [1000, 500, 250]
    n_neurons_of_hidden_layer_x_discriminator = [500, 250, 125]

    n_epochs = 5
    learning_rate = 0.001
    beta1 = 0.9
    results_path = './Results'

    # loss functions
    loss_functions = ["hinge_loss",
                      "mean_squared_error",
                      "sigmoid_cross_entropy",
                      "softmax_cross_entropy"]

    loss_function_discriminator = "sigmoid_cross_entropy"
    loss_function_generator = "hinge_loss"

    print("loss function discr:" + loss_function_discriminator)
    print("loss function gen:" + loss_function_generator)
    print("")

    adv_autoencoder = AdversarialAutoencoder. \
        AdversarialAutoencoder(input_dim=input_dim,
                               n_neurons_of_hidden_layer_x_autoencoder=n_neurons_of_hidden_layer_x_autoencoder,
                               n_neurons_of_hidden_layer_x_discriminator=n_neurons_of_hidden_layer_x_discriminator,
                               z_dim=z_dim, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
                               beta1=beta1, loss_function_discriminator=loss_function_discriminator,
                               loss_function_generator=loss_function_generator, results_path=results_path)

    adv_autoencoder.train(True)


if __name__ == '__main__':
    testing()
