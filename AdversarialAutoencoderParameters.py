import random


class AdversarialAutoencoderParameters:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def update_parameter(self, parameter_name, parameter_value):
        if parameter_name in self.parameters:
            self.parameters[parameter_name] = parameter_value
            print(self.parameters[parameter_name])

    def get_single_parameter(self, parameter_name):
        if parameter_name in self.parameters:
            print(self.parameters[parameter_name])
            return self.parameters[parameter_name]

    def get_all_parameters(self):
        print(self.parameters)
        return self.parameters

    @staticmethod
    def get_default_parameters():
        return {'batch_size': 100, 'n_epochs': 10,
                'n_neurons_of_hidden_layer_x_autoencoder': [1000, 500, 250],
                'n_neurons_of_hidden_layer_x_discriminator': [500, 250, 125],
                'bias_init_value_of_hidden_layer_x_autoencoder': [0.0, 0.0, 0.0, 0.0],
                'bias_init_value_of_hidden_layer_x_discriminator': [0.0, 0.0, 0.0, 0.0],
                'learning_rate_autoencoder': 0.001, 'learning_rate_discriminator': 0.01,
                'learning_rate_generator': 0.01, 'optimizer_autoencoder': 'AdamOptimizer',
                'optimizer_discriminator': 'RMSPropOptimizer', 'optimizer_generator': 'RMSPropOptimizer',
                'AdadeltaOptimizer_rho_autoencoder': 0.95, 'AdadeltaOptimizer_epsilon_autoencoder': 1e-08,
                'AdadeltaOptimizer_rho_discriminator': 0.95, 'AdadeltaOptimizer_epsilon_discriminator': 1e-08,
                'AdadeltaOptimizer_rho_generator': 0.95, 'AdadeltaOptimizer_epsilon_generator': 1e-08,
                'AdagradOptimizer_initial_accumulator_value_autoencoder': 0.1,
                'AdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
                'AdagradOptimizer_initial_accumulator_value_generator': 0.1,
                'MomentumOptimizer_momentum_autoencoder': 0.9, 'MomentumOptimizer_use_nesterov_autoencoder': False,
                'MomentumOptimizer_momentum_discriminator': 0.9, 'MomentumOptimizer_use_nesterov_discriminator': False,
                'MomentumOptimizer_momentum_generator': 0.9, 'MomentumOptimizer_use_nesterov_generator': False,
                'AdamOptimizer_beta1_autoencoder': 0.9, 'AdamOptimizer_beta2_autoencoder': 0.999,
                'AdamOptimizer_epsilon_autoencoder': 1e-08, 'AdamOptimizer_beta1_discriminator': 0.9,
                'AdamOptimizer_beta2_discriminator': 0.999, 'AdamOptimizer_epsilon_discriminator': 1e-08,
                'AdamOptimizer_beta1_generator': 0.9, 'AdamOptimizer_beta2_generator': 0.999,
                'AdamOptimizer_epsilon_generator': 1e-08, 'FtrlOptimizer_learning_rate_power_autoencoder': -0.5,
                'FtrlOptimizer_initial_accumulator_value_autoencoder': 0.1,
                'FtrlOptimizer_l1_regularization_strength_autoencoder': 0.0,
                'FtrlOptimizer_l2_regularization_strength_autoencoder': 0.0,
                'FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder': 0.0,
                'FtrlOptimizer_learning_rate_power_discriminator': -0.5,
                'FtrlOptimizer_initial_accumulator_value_discriminator': 0.1,
                'FtrlOptimizer_l1_regularization_strength_discriminator': 0.0,
                'FtrlOptimizer_l2_regularization_strength_discriminator': 0.0,
                'FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator': 0.0,
                'FtrlOptimizer_learning_rate_power_generator': -0.5,
                'FtrlOptimizer_initial_accumulator_value_generator': 0.1,
                'FtrlOptimizer_l1_regularization_strength_generator': 0.0,
                'FtrlOptimizer_l2_regularization_strength_generator': 0.0,
                'FtrlOptimizer_l2_shrinkage_regularization_strength_generator': 0.0,
                'ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder': 0.0,
                'ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder': 0.0,
                'ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator': 0.0,
                'ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator': 0.0,
                'ProximalGradientDescentOptimizer_l1_regularization_strength_generator': 0.0,
                'ProximalGradientDescentOptimizer_l2_regularization_strength_generator': 0.0,
                'ProximalAdagradOptimizer_initial_accumulator_value_autoencoder': 0.2,
                'ProximalAdagradOptimizer_l1_regularization_strength_autoencoder': 0.0,
                'ProximalAdagradOptimizer_l2_regularization_strength_autoencoder': 0.0,
                'ProximalAdagradOptimizer_initial_accumulator_value_discriminator': 0.1,
                'ProximalAdagradOptimizer_l1_regularization_strength_discriminator': 0.0,
                'ProximalAdagradOptimizer_l2_regularization_strength_discriminator': 0.0,
                'ProximalAdagradOptimizer_initial_accumulator_value_generator': 0.1,
                'ProximalAdagradOptimizer_l1_regularization_strength_generator': 0.0,
                'ProximalAdagradOptimizer_l2_regularization_strength_generator': 0.0,
                'RMSPropOptimizer_decay_autoencoder': 0.9, 'RMSPropOptimizer_momentum_autoencoder': 0.0,
                'RMSPropOptimizer_epsilon_autoencoder': 1e-10, 'RMSPropOptimizer_centered_autoencoder': False,
                'RMSPropOptimizer_decay_discriminator': 0.9, 'RMSPropOptimizer_momentum_discriminator': 0.0,
                'RMSPropOptimizer_epsilon_discriminator': 1e-10, 'RMSPropOptimizer_centered_discriminator': False,
                'RMSPropOptimizer_decay_generator': 0.9, 'RMSPropOptimizer_momentum_generator': 0.0,
                'RMSPropOptimizer_epsilon_generator': 1e-10, 'RMSPropOptimizer_centered_generator': False,
                'loss_function_discriminator': 'sigmoid_cross_entropy', 'loss_function_generator': 'hinge_loss',
                'results_path': './Results'}

    def get_randomized_parameters(self, *args):
        """
        returns randomized values for the specified parameters; otherwise the default values
        :param args: string or list of strings with the parameters which should be randomized; if empty randomizes
        all parameters
        :return: dictionary: {'parameter1': parameter1_value, 'parameter2': parameter2_value}
        """

        # TODO: maybe implement a separate function to randomize the more important parameters such as batch_size,
        # the network structure, etc.

        # train duration
        batch_size = random.randint(50, 500)
        n_epochs = 10  # TODO: probably doesn't make really sense ..

        # network topology
        n_neurons_of_hidden_layer_x_autoencoder = [1000, 500, 250]  # TODO: function to create a random number of layers
        n_neurons_of_hidden_layer_x_discriminator = [500, 250, 125]  # TODO: and neurons; should be smaller to the end

        # initial bias values for the hidden layers
        bias_init_value_of_hidden_layer_x_autoencoder = [0.0, 0.0, 0.0,
                                                         0.0]  # TODO: function to create a random number of layers
        bias_init_value_of_hidden_layer_x_discriminator = [0.0, 0.0, 0.0,
                                                           0.0]  # TODO: and neurons; should be smaller to the end

        # individual learning rates
        learning_rate_autoencoder = random.uniform(0.0001, 0.1)
        learning_rate_discriminator = random.uniform(0.0001, 0.1)
        learning_rate_generator = random.uniform(0.0001, 0.1)

        # available optimizers:
        autoencoder_optimizers = ["AdamOptimizer",
                                  "RMSPropOptimizer"]

        optimizers = ["GradientDescentOptimizer",  # autoencoder part not working
                      "AdadeltaOptimizer",  # autoencoder part not working
                      "AdagradOptimizer",  # autoencoder part not working
                      "MomentumOptimizer",  # autoencoder part not working
                      "AdamOptimizer",
                      # "FtrlOptimizer",  # autoencoder part not working; optimizer slow + bad results
                      "ProximalGradientDescentOptimizer",  # autoencoder part not working
                      "ProximalAdagradOptimizer",  # autoencoder part not working
                      "RMSPropOptimizer"]

        optimizer_autoencoder = random.choice(autoencoder_optimizers)
        optimizer_discriminator = random.choice(optimizers)
        optimizer_generator = random.choice(optimizers)

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
        AdadeltaOptimizer_rho_autoencoder = random.uniform(0.8, 0.99)
        AdadeltaOptimizer_epsilon_autoencoder = random.uniform(1e-09, 1e-07)
        AdadeltaOptimizer_rho_discriminator = random.uniform(0.8, 0.99)
        AdadeltaOptimizer_epsilon_discriminator = random.uniform(1e-09, 1e-07)
        AdadeltaOptimizer_rho_generator = random.uniform(0.8, 0.99)
        AdadeltaOptimizer_epsilon_generator = random.uniform(1e-09, 1e-07)

        # AdagradOptimizer
        #   - learning rate
        #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
        #   default: 0.1
        AdagradOptimizer_initial_accumulator_value_autoencoder = random.uniform(0.01, 0.2)
        AdagradOptimizer_initial_accumulator_value_discriminator = random.uniform(0.01, 0.2)
        AdagradOptimizer_initial_accumulator_value_generator = random.uniform(0.01, 0.2)

        # MomentumOptimizer
        #   - learning rate
        #   - momentum: A Tensor or a floating point value. The momentum.
        #   - use_nesterov: If True use Nesterov Momentum; default: False http://proceedings.mlr.press/v28/sutskever13.pdf
        MomentumOptimizer_momentum_autoencoder = random.uniform(0.8, 1.0)
        MomentumOptimizer_use_nesterov_autoencoder = random.choice([True, False])
        MomentumOptimizer_momentum_discriminator = random.uniform(0.8, 1.0)
        MomentumOptimizer_use_nesterov_discriminator = random.choice([True, False])
        MomentumOptimizer_momentum_generator = random.uniform(0.8, 1.0)
        MomentumOptimizer_use_nesterov_generator = random.choice([True, False])

        # AdamOptimizer
        #   - learning rate; default: 0.001
        #   - beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
        #   default: 0.9
        #   - beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
        #   default: 0.999
        #   - epsilon: A small constant for numerical stability. default: 1e-08
        AdamOptimizer_beta1_autoencoder =  random.uniform(0.8, 1.0)
        AdamOptimizer_beta2_autoencoder =  random.uniform(0.99, 1.0)
        AdamOptimizer_epsilon_autoencoder = random.uniform(1e-07, 1e-09)
        AdamOptimizer_beta1_discriminator = random.uniform(0.8, 1.0)
        AdamOptimizer_beta2_discriminator = random.uniform(0.99, 1.0)
        AdamOptimizer_epsilon_discriminator = random.uniform(1e-07, 1e-09)
        AdamOptimizer_beta1_generator = random.uniform(0.8, 1.0)
        AdamOptimizer_beta2_generator = random.uniform(0.99, 1.0)
        AdamOptimizer_epsilon_generator = random.uniform(1e-07, 1e-09)

        # FtrlOptimizer
        #   - learning rate
        #   - learning rate power: A float value, must be less or equal to zero. default: -0.5
        #   - initial_accumulator_value: The starting value for accumulators. Only positive values are allowed. default: 0.1
        #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        #   - l2_shrinkage_regularization_strength: A float value, must be greater than or equal to zero. This differs from
        #   L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
        #   default: 0.0
        FtrlOptimizer_learning_rate_power_autoencoder = random.uniform(-1, 0)
        FtrlOptimizer_initial_accumulator_value_autoencoder = random.uniform(0.0, 0.2)
        FtrlOptimizer_l1_regularization_strength_autoencoder = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_regularization_strength_autoencoder = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_shrinkage_regularization_strength_autoencoder = random.uniform(0.0, 0.1)
        FtrlOptimizer_learning_rate_power_discriminator = random.uniform(-1, 0)
        FtrlOptimizer_initial_accumulator_value_discriminator = random.uniform(0.0, 0.2)
        FtrlOptimizer_l1_regularization_strength_discriminator = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_regularization_strength_discriminator = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_shrinkage_regularization_strength_discriminator = random.uniform(0.0, 0.1)
        FtrlOptimizer_learning_rate_power_generator = random.uniform(-1, 0)
        FtrlOptimizer_initial_accumulator_value_generator = random.uniform(0.0, 0.2)
        FtrlOptimizer_l1_regularization_strength_generator = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_regularization_strength_generator = random.uniform(0.0, 0.1)
        FtrlOptimizer_l2_shrinkage_regularization_strength_generator = random.uniform(0.0, 0.1)

        # ProximalGradientDescentOptimizer
        #   - learning rate
        #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        ProximalGradientDescentOptimizer_l1_regularization_strength_autoencoder = random.uniform(0.0, 0.2)
        ProximalGradientDescentOptimizer_l2_regularization_strength_autoencoder = random.uniform(0.0, 0.1)
        ProximalGradientDescentOptimizer_l1_regularization_strength_discriminator = random.uniform(0.0, 0.2)
        ProximalGradientDescentOptimizer_l2_regularization_strength_discriminator = random.uniform(0.0, 0.1)
        ProximalGradientDescentOptimizer_l1_regularization_strength_generator = random.uniform(0.0, 0.2)
        ProximalGradientDescentOptimizer_l2_regularization_strength_generator = random.uniform(0.0, 0.1)

        # ProximalAdagradOptimizer
        #   - learning rate
        #   - initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
        #   default: 0.1
        #   - l1_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        #   - l2_regularization_strength: A float value, must be greater than or equal to zero. default: 0.0
        ProximalAdagradOptimizer_initial_accumulator_value_autoencoder = random.uniform(0.01, 0.3)
        ProximalAdagradOptimizer_l1_regularization_strength_autoencoder = random.uniform(0.0, 0.2)
        ProximalAdagradOptimizer_l2_regularization_strength_autoencoder = random.uniform(0.0, 0.1)
        ProximalAdagradOptimizer_initial_accumulator_value_discriminator = random.uniform(0.01, 0.3)
        ProximalAdagradOptimizer_l1_regularization_strength_discriminator = random.uniform(0.0, 0.2)
        ProximalAdagradOptimizer_l2_regularization_strength_discriminator = random.uniform(0.0, 0.1)
        ProximalAdagradOptimizer_initial_accumulator_value_generator = random.uniform(0.01, 0.3)
        ProximalAdagradOptimizer_l1_regularization_strength_generator = random.uniform(0.0, 0.2)
        ProximalAdagradOptimizer_l2_regularization_strength_generator = random.uniform(0.0, 0.1)

        # RMSPropOptimizer
        #   - learning rate
        #   - decay: Discounting factor for the history/coming gradient; default: 0.9
        #   - momentum: A scalar tensor; default: 0.0.
        #   - epsilon:  Small value to avoid zero denominator.; default: 1e-10
        #   - centered: If True, gradients are normalized by the estimated variance of the gradient; if False, by the
        #   uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms
        #   of computation and memory. Defaults to False.
        RMSPropOptimizer_decay_autoencoder = random.uniform(0.8, 1.0)
        RMSPropOptimizer_momentum_autoencoder = random.uniform(0.0, 0.2)
        RMSPropOptimizer_epsilon_autoencoder = random.uniform(1e-9, 1e-11)
        RMSPropOptimizer_centered_autoencoder = random.choice([True, False])
        RMSPropOptimizer_decay_discriminator = random.uniform(0.8, 1.0)
        RMSPropOptimizer_momentum_discriminator = random.uniform(0.0, 0.2)
        RMSPropOptimizer_epsilon_discriminator = random.uniform(1e-9, 1e-11)
        RMSPropOptimizer_centered_discriminator = random.choice([True, False])
        RMSPropOptimizer_decay_generator = random.uniform(0.8, 1.0)
        RMSPropOptimizer_momentum_generator = random.uniform(0.0, 0.2)
        RMSPropOptimizer_epsilon_generator = random.uniform(1e-9, 1e-11)
        RMSPropOptimizer_centered_generator = random.choice([True, False])

        # available loss functions
        loss_functions = ["hinge_loss",
                          "mean_squared_error",
                          "sigmoid_cross_entropy",
                          "softmax_cross_entropy"]

        # loss function for discriminator
        loss_function_discriminator = random.choice(loss_functions)
        # loss function for generator
        loss_function_generator = random.choice(loss_functions)

        # get the default parameters
        param_dict = self.get_default_parameters()

        # iterate over the variable names provided as parameters and set their value as random defined above
        if args:
            for var_name in args:
                param_dict[var_name] = locals()[var_name]
        else:
            local_vars_to_ignore = ["loss_functions", "param_dict", "optimizer", "autoencoder_optimizers"]
            for var_name in list(locals()): # convert to list to avoid RuntimeError: dictionary changed during iteration
                if not var_name in local_vars_to_ignore:
                    param_dict[var_name] = locals()[var_name]

        # print(param_dict)
        return param_dict
