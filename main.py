import AdversarialAutoencoder


def testing():

    # input data parameters
    input_dim = 784
    z_dim = 2

    # train duration
    batch_size = 100
    n_epochs = 10

    # network topology
    n_neurons_of_hidden_layer_x_autoencoder = [1000, 500, 250]
    n_neurons_of_hidden_layer_x_discriminator = [500, 250, 125]

    # initial bias values for the hidden layers
    bias_init_value_of_hidden_layer_x_autoencoder = [0.0, 0.0, 0.0, 0.0]
    bias_init_value_of_hidden_layer_x_discriminator = [0.0, 0.0, 0.0, 0.0]

    # individual learning rates
    learning_rate_autoencoder = 0.001
    learning_rate_discriminator = 0.001
    learning_rate_generator = 0.001

    # exponential decay rate for the 1st moment estimates for the adam optimizer.
    beta1_autoencoder = 0.9
    beta1_discriminator = 0.9
    beta1_generator = 0.9

    # exponential decay rate for the 2nd moment estimates for the adam optimizer.
    beta2_autoencoder = 0.999
    beta2_discriminator = 0.999
    beta2_generator = 0.999

    # TODO: other optimizers
    # https://www.tensorflow.org/api_guides/python/train#Optimizers

    # path to the results
    results_path = './Results'

    # available loss functions
    loss_functions = ["hinge_loss",
                      "mean_squared_error",
                      "sigmoid_cross_entropy",
                      "softmax_cross_entropy"]

    # loss function for discriminator
    loss_function_discriminator = "sigmoid_cross_entropy"
    # loss function for generator
    loss_function_generator = "hinge_loss"

    print("loss function discr:" + loss_function_discriminator)
    print("loss function gen:" + loss_function_generator)
    print("")

    adv_autoencoder = AdversarialAutoencoder. \
        AdversarialAutoencoder(input_dim=input_dim, z_dim=z_dim, batch_size=batch_size, n_epochs=n_epochs,
                               n_neurons_of_hidden_layer_x_autoencoder=n_neurons_of_hidden_layer_x_autoencoder,
                               n_neurons_of_hidden_layer_x_discriminator=n_neurons_of_hidden_layer_x_discriminator,
                               bias_init_value_of_hidden_layer_x_autoencoder=
                                                     bias_init_value_of_hidden_layer_x_autoencoder,
                               bias_init_value_of_hidden_layer_x_discriminator=
                                                     bias_init_value_of_hidden_layer_x_discriminator,

                               learning_rate_autoencoder=learning_rate_autoencoder,
                               learning_rate_discriminator=learning_rate_discriminator,
                               learning_rate_generator=learning_rate_generator,

                               beta1_autoencoder=beta1_autoencoder,
                               beta1_discriminator=beta1_discriminator,
                               beta1_generator=beta1_generator,

                               beta2_autoencoder=beta2_autoencoder,
                               beta2_discriminator=beta2_discriminator,
                               beta2_generator=beta2_generator,

                               loss_function_discriminator=loss_function_discriminator,
                               loss_function_generator=loss_function_generator, results_path=results_path)

    adv_autoencoder.train(True)


if __name__ == '__main__':
    testing()
