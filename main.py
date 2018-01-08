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
    learning_rate_discriminator = 0.01
    learning_rate_generator = 0.01

    # available optimizers:
    optimizers = ["GradientDescentOptimizer",           # autoencoder part not working
                  "AdadeltaOptimizer",                  # autoencoder part not working
                  "AdagradOptimizer",                   # autoencoder part not working
                  "MomentumOptimizer",                  # autoencoder part not working
                  "AdamOptimizer",
                  "FtrlOptimizer",                      # autoencoder part not working; optimizer slow + bad results
                  "ProximalGradientDescentOptimizer",   # autoencoder part not working
                  "ProximalAdagradOptimizer",           # autoencoder part not working
                  "RMSPropOptimizer"]

    optimizer_autoencoder = "AdamOptimizer"
    optimizer_discriminator = "RMSPropOptimizer"
    optimizer_generator = "RMSPropOptimizer"

    """
    https://www.tensorflow.org/api_guides/python/train#Optimizers
    parameters for optimizers:
    """
    # GradientDescentOptimizer:
    #   - learning rate:

    # AdadeltaOptimizer():
    #   - learning rate: default: 0.01
    #   - rho: decay rate; default: 0.95
    #   - epsilon: A constant epsilon used to better conditioning the grad update; default: 1e-08
    AdadeltaOptimizer_rho_autoencoder = 0.95
    AdadeltaOptimizer_epsilon_autoencoder = 1e-08

    AdadeltaOptimizer_rho_discriminator = 0.95
    AdadeltaOptimizer_epsilon_discriminator = 1e-08

    AdadeltaOptimizer_rho_generator = 0.95
    AdadeltaOptimizer_epsilon_generator = 1e-08

    # AdagradOptimizer
    #   - learning rate
    #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
    #   default: 0.1
    AdagradOptimizer_initial_accumulator_value_autoencoder = 0.1

    AdagradOptimizer_initial_accumulator_value_discriminator = 0.1

    AdagradOptimizer_initial_accumulator_value_generator = 0.1

    # MomentumOptimizer
    #   - learning rate
    #   - momentum: A Tensor or a floating point value. The momentum.
    #   - use_nesterov: If True use Nesterov Momentum; default: False http://proceedings.mlr.press/v28/sutskever13.pdf
    MomentumOptimizer_momentum_autoencoder = 0.9
    MomentumOptimizer_use_nesterov_autoencoder = False

    MomentumOptimizer_momentum_discriminator = 0.9
    MomentumOptimizer_use_nesterov_discriminator = False

    MomentumOptimizer_momentum_generator = 0.9
    MomentumOptimizer_use_nesterov_generator = False

    # AdamOptimizer
    #   - learning rate; default: 0.001
    #   - beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
    #   default: 0.9
    #   - beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
    #   default: 0.99
    #   - epsilon: A small constant for numerical stability. default: 1e-08
    AdamOptimizer_beta1_autoencoder = 0.9
    AdamOptimizer_beta2_autoencoder = 0.999
    AdamOptimizer_epsilon_autoencoder = 1e-08

    AdamOptimizer_beta1_discriminator = 0.9
    AdamOptimizer_beta2_discriminator = 0.999
    AdamOptimizer_epsilon_discriminator = 1e-08

    AdamOptimizer_beta1_generator = 0.9
    AdamOptimizer_beta2_generator = 0.999
    AdamOptimizer_epsilon_generator = 1e-08

    # FtrlOptimizer
    #   - learning rate
    #   - learning rate power: A float value, must be less or equal to zero. default: -0.5
    #   - initial_accumulator_value: The starting value for accumulators. Only positive values are allowed. default: 0.1
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_shrinkage_regularization_strength: A float value, must be greater than or equal to zero. This differs from
    #   L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
    #   default: 0.0
    FtrlOptimizer_learning_rate_power_autoencoder = -0.5
    FtrlOptimizer_initial_accumulator_value_autoencoder = 0.1
    FtrlOptimizer_l1_regularization_strength_autoencoder = 0.0
    FtrlOptimizer_l2_regularization_strength_autoencoder = 0.0
    FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder = 0.0

    FtrlOptimizer_learning_rate_power_discriminator = -0.5
    FtrlOptimizer_initial_accumulator_value_discriminator = 0.1
    FtrlOptimizer_l1_regularization_strength_discriminator = 0.0
    FtrlOptimizer_l2_regularization_strength_discriminator = 0.0
    FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator = 0.0

    FtrlOptimizer_learning_rate_power_generator = -0.5
    FtrlOptimizer_initial_accumulator_value_generator = 0.1
    FtrlOptimizer_l1_regularization_strength_generator = 0.0
    FtrlOptimizer_l2_regularization_strength_generator = 0.0
    FtrlOptimizer_l2_shrinkage_regularization_strength_generator = 0.0

    # ProximalGradientDescentOptimizer
    #   - learning rate
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder = 0.0
    ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder = 0.0

    ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator = 0.0
    ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator = 0.0

    ProximalGradientDescentOptimizer_l1_regularization_strength_generator = 0.0
    ProximalGradientDescentOptimizer_l2_regularization_strength_generator = 0.0

    # ProximalAdagradOptimizer
    #   - learning rate
    #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
    #   default: 0.1
    #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
    ProximalAdagradOptimizer_initial_accumulator_value_autoencoder = 0.1
    ProximalAdagradOptimizer_l1_regularization_strength_autoencoder = 0.0
    ProximalAdagradOptimizer_l2_regularization_strength_autoencoder = 0.0

    ProximalAdagradOptimizer_initial_accumulator_value_discriminator = 0.1
    ProximalAdagradOptimizer_l1_regularization_strength_discriminator = 0.0
    ProximalAdagradOptimizer_l2_regularization_strength_discriminator = 0.0

    ProximalAdagradOptimizer_initial_accumulator_value_generator = 0.1
    ProximalAdagradOptimizer_l1_regularization_strength_generator = 0.0
    ProximalAdagradOptimizer_l2_regularization_strength_generator = 0.0

    # RMSPropOptimizer
    #   - learning rate
    #   - decay: Discounting factor for the history/coming gradient; default: 0.9
    #   - momentum: A scalar tensor; default: 0.0.
    #   - epsilon:  Small value to avoid zero denominator.; default: 1e-10
    #   - centered: If True, gradients are normalized by the estimated variance of the gradient; if False, by the
    #   uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms
    #   of computation and memory. Defaults to False.
    RMSPropOptimizer_decay_autoencoder = 0.9
    RMSPropOptimizer_momentum_autoencoder = 0.0
    RMSPropOptimizer_epsilon_autoencoder = 1e-10
    RMSPropOptimizer_centered_autoencoder = False

    RMSPropOptimizer_decay_discriminator = 0.9
    RMSPropOptimizer_momentum_discriminator = 0.0
    RMSPropOptimizer_epsilon_discriminator = 1e-10
    RMSPropOptimizer_centered_discriminator = False

    RMSPropOptimizer_decay_generator = 0.9
    RMSPropOptimizer_momentum_generator = 0.0
    RMSPropOptimizer_epsilon_generator = 1e-10
    RMSPropOptimizer_centered_generator = False

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

                               optimizer_autoencoder=optimizer_autoencoder,
                               optimizer_discriminator=optimizer_discriminator,
                               optimizer_generator=optimizer_generator,


                               AdadeltaOptimizer_rho_autoencoder=AdadeltaOptimizer_rho_autoencoder,
                               AdadeltaOptimizer_epsilon_autoencoder=AdadeltaOptimizer_epsilon_autoencoder,

                               AdadeltaOptimizer_rho_discriminator=AdadeltaOptimizer_rho_discriminator,
                               AdadeltaOptimizer_epsilon_discriminator=AdadeltaOptimizer_epsilon_discriminator,

                               AdadeltaOptimizer_rho_generator=AdadeltaOptimizer_rho_generator,
                               AdadeltaOptimizer_epsilon_generator=AdadeltaOptimizer_epsilon_generator,


                               AdagradOptimizer_initial_accumulator_value_autoencoder=
                                                     AdagradOptimizer_initial_accumulator_value_autoencoder,
                               AdagradOptimizer_initial_accumulator_value_discriminator=
                                                     AdagradOptimizer_initial_accumulator_value_discriminator,
                               AdagradOptimizer_initial_accumulator_value_generator=
                                                     AdagradOptimizer_initial_accumulator_value_generator,


                               MomentumOptimizer_momentum_autoencoder=MomentumOptimizer_momentum_autoencoder,
                               MomentumOptimizer_use_nesterov_autoencoder=MomentumOptimizer_use_nesterov_autoencoder,

                               MomentumOptimizer_momentum_discriminator=MomentumOptimizer_momentum_discriminator,
                               MomentumOptimizer_use_nesterov_discriminator=MomentumOptimizer_use_nesterov_discriminator,

                               MomentumOptimizer_momentum_generator=MomentumOptimizer_momentum_generator,
                               MomentumOptimizer_use_nesterov_generator=MomentumOptimizer_use_nesterov_generator,


                               AdamOptimizer_beta1_autoencoder=AdamOptimizer_beta1_autoencoder,
                               AdamOptimizer_beta2_autoencoder=AdamOptimizer_beta2_autoencoder,
                               AdamOptimizer_epsilon_autoencoder=AdamOptimizer_epsilon_autoencoder,

                               AdamOptimizer_beta1_discriminator=AdamOptimizer_beta1_discriminator,
                               AdamOptimizer_beta2_discriminator=AdamOptimizer_beta2_discriminator,
                               AdamOptimizer_epsilon_discriminator=AdamOptimizer_epsilon_discriminator,

                               AdamOptimizer_beta1_generator=AdamOptimizer_beta1_generator,
                               AdamOptimizer_beta2_generator=AdamOptimizer_beta2_generator,
                               AdamOptimizer_epsilon_generator=AdamOptimizer_epsilon_generator,


                               FtrlOptimizer_learning_rate_power_autoencoder=
                                                     FtrlOptimizer_learning_rate_power_autoencoder,
                               FtrlOptimizer_initial_accumulator_value_autoencoder=
                                                     FtrlOptimizer_initial_accumulator_value_autoencoder,
                               FtrlOptimizer_l1_regularization_strength_autoencoder=
                                                     FtrlOptimizer_l1_regularization_strength_autoencoder,
                               FtrlOptimizer_l2_regularization_strength_autoencoder=
                                                     FtrlOptimizer_l2_regularization_strength_autoencoder,
                               FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder=
                                                     FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder,

                               FtrlOptimizer_learning_rate_power_discriminator=
                                                     FtrlOptimizer_learning_rate_power_discriminator,
                               FtrlOptimizer_initial_accumulator_value_discriminator=
                                                     FtrlOptimizer_initial_accumulator_value_discriminator,
                               FtrlOptimizer_l1_regularization_strength_discriminator=
                                                     FtrlOptimizer_l1_regularization_strength_discriminator,
                               FtrlOptimizer_l2_regularization_strength_discriminator=
                                                     FtrlOptimizer_l2_regularization_strength_discriminator,
                               FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator
                                                     =FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator,

                               FtrlOptimizer_learning_rate_power_generator=
                                                     FtrlOptimizer_learning_rate_power_generator,
                               FtrlOptimizer_initial_accumulator_value_generator=
                                                     FtrlOptimizer_initial_accumulator_value_generator,
                               FtrlOptimizer_l1_regularization_strength_generator=
                                                     FtrlOptimizer_l1_regularization_strength_generator,
                               FtrlOptimizer_l2_regularization_strength_generator=
                                                     FtrlOptimizer_l2_regularization_strength_generator,
                               FtrlOptimizer_l2_shrinkage_regularization_strength_generator=
                                                     FtrlOptimizer_l2_shrinkage_regularization_strength_generator,


                               ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder=
                                                     ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder,
                               ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder=
                                                     ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder,

                               ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator=
                                                     ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator,
                               ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator=
                                                     ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator,

                               ProximalGradientDescentOptimizer_l1_regularization_strength_generator=
                                                     ProximalGradientDescentOptimizer_l1_regularization_strength_generator,
                               ProximalGradientDescentOptimizer_l2_regularization_strength_generator=
                                                     ProximalGradientDescentOptimizer_l2_regularization_strength_generator,


                               ProximalAdagradOptimizer_initial_accumulator_value_autoencoder=
                                                     ProximalAdagradOptimizer_initial_accumulator_value_autoencoder,
                               ProximalAdagradOptimizer_l1_regularization_strength_autoencoder=
                                                     ProximalAdagradOptimizer_l1_regularization_strength_autoencoder,
                               ProximalAdagradOptimizer_l2_regularization_strength_autoencoder=
                                                     ProximalAdagradOptimizer_l2_regularization_strength_autoencoder,

                               ProximalAdagradOptimizer_initial_accumulator_value_discriminator=
                                                     ProximalAdagradOptimizer_initial_accumulator_value_discriminator,
                               ProximalAdagradOptimizer_l1_regularization_strength_discriminator=
                                                     ProximalAdagradOptimizer_l1_regularization_strength_discriminator,
                               ProximalAdagradOptimizer_l2_regularization_strength_discriminator=
                                                     ProximalAdagradOptimizer_l2_regularization_strength_discriminator,

                               ProximalAdagradOptimizer_initial_accumulator_value_generator=
                                                     ProximalAdagradOptimizer_initial_accumulator_value_generator,
                               ProximalAdagradOptimizer_l1_regularization_strength_generator=
                                                     ProximalAdagradOptimizer_l1_regularization_strength_generator,
                               ProximalAdagradOptimizer_l2_regularization_strength_generator=
                                                     ProximalAdagradOptimizer_l2_regularization_strength_generator,


                               RMSPropOptimizer_decay_autoencoder=RMSPropOptimizer_decay_autoencoder,
                               RMSPropOptimizer_momentum_autoencoder=RMSPropOptimizer_momentum_autoencoder,
                               RMSPropOptimizer_epsilon_autoencoder=RMSPropOptimizer_epsilon_autoencoder,
                               RMSPropOptimizer_centered_autoencoder=RMSPropOptimizer_centered_autoencoder,

                               RMSPropOptimizer_decay_discriminator=RMSPropOptimizer_decay_discriminator,
                               RMSPropOptimizer_momentum_discriminator=RMSPropOptimizer_momentum_discriminator,
                               RMSPropOptimizer_epsilon_discriminator=RMSPropOptimizer_epsilon_discriminator,
                               RMSPropOptimizer_centered_discriminator=RMSPropOptimizer_centered_discriminator,

                               RMSPropOptimizer_decay_generator=RMSPropOptimizer_decay_generator,
                               RMSPropOptimizer_momentum_generator=RMSPropOptimizer_momentum_generator,
                               RMSPropOptimizer_epsilon_generator=RMSPropOptimizer_epsilon_generator,
                               RMSPropOptimizer_centered_generator=RMSPropOptimizer_centered_generator,


                               loss_function_discriminator=loss_function_discriminator,
                               loss_function_generator=loss_function_generator, results_path=results_path)

    adv_autoencoder.train(True)


if __name__ == '__main__':
    testing()
