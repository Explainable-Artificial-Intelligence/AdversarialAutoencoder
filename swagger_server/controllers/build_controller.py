import connexion
import sys

from autoencoders.DimensionalityReductionAdversarialAutoencoder import DimensionalityReductionAdversarialAutoencoder
from autoencoders.IncorporatingLabelInformationAdversarialAutoencoder import \
    IncorporatingLabelInformationAdversarialAutoencoder
from autoencoders.UnsupervisedClusteringAdversarialAutoencoder import UnsupervisedClusteringAdversarialAutoencoder
from swagger_server.utils.Storage import Storage

from autoencoders.SemiSupervisedAdversarialAutoencoder import SemiSupervisedAdversarialAutoencoder
from autoencoders.SupervisedAdversarialAutoencoder import SupervisedAdversarialAutoencoder
from autoencoders.UnsupervisedAdversarialAutoencoder import UnsupervisedAdversarialAutoencoder
from util.AdversarialAutoencoderParameters import get_result_path_for_selected_autoencoder


def build_aae(selected_autoencoder, aae_parameters):
    """
    builds the adversarial autoencoder with the parameters provided
    :param selected_autoencoder: one of ["Unsupervised", "Supervised", "SemiSupervised"]
    :param aae_parameters: parameters for the adv. autoencoder
    :return:
    """

    if connexion.request.is_json:

        # get the parameters for the adv autoencoder
        aae_parameters = connexion.request.get_json()

        # check if we have a dataset selected
        if not Storage.get_selected_dataset():
            return "Error: data set not found", 404

        # get the selected dataset ["MNIST", "SVHN", "cifar10", "custom"]
        selected_dataset = Storage.get_selected_dataset()
        aae_parameters["selected_dataset"] = selected_dataset

        # get the results_path based on the selected autoencoder
        aae_parameters["results_path"] = get_result_path_for_selected_autoencoder(selected_autoencoder)

        # set the input dim and the color scale according to the selected dataset
        if selected_dataset == "MNIST":
            aae_parameters["input_dim_x"] = 28
            aae_parameters["input_dim_y"] = 28
            aae_parameters["color_scale"] = "gray_scale"
        elif selected_dataset == "SVHN":
            aae_parameters["input_dim_x"] = 32
            aae_parameters["input_dim_y"] = 32
            aae_parameters["color_scale"] = "rgb_scale"
        elif selected_dataset == "cifar10":
            aae_parameters["input_dim_x"] = 32
            aae_parameters["input_dim_y"] = 32
            aae_parameters["color_scale"] = "rgb_scale"
        elif selected_dataset == "custom":
            return "Error: not implemented", 404

        if Storage.get_aae() is not None:
            # reset the tensorflow graph
            Storage.get_aae().reset_graph()

        # create the AAE with the current parameters
        adv_autoencoder = None
        try:
            if selected_autoencoder == "Unsupervised":
                adv_autoencoder = UnsupervisedAdversarialAutoencoder(aae_parameters)
            elif selected_autoencoder == "Supervised":
                adv_autoencoder = SupervisedAdversarialAutoencoder(aae_parameters)
            elif selected_autoencoder == "SemiSupervised":
                adv_autoencoder = SemiSupervisedAdversarialAutoencoder(aae_parameters)
            elif selected_autoencoder == "IncorporatingLabelInformation":
                adv_autoencoder = IncorporatingLabelInformationAdversarialAutoencoder(aae_parameters)
            elif selected_autoencoder == "UnsupervisedClustering":
                adv_autoencoder = UnsupervisedClusteringAdversarialAutoencoder(aae_parameters)
            elif selected_autoencoder == "DimensionalityReduction":
                adv_autoencoder = DimensionalityReductionAdversarialAutoencoder(aae_parameters)
        except KeyError:
            return 'Error: Parameter %s not found' % sys.exc_info()[1], 404

        # store the parameters and the adv. autoencoder in the storage class
        Storage.set_aae(adv_autoencoder)
        Storage.set_aae_parameters(aae_parameters)
        Storage.set_selected_autoencoder(selected_autoencoder)

        return "Success: AAE successfully built", 200
    return 'Error: parameters not in .json format', 415

