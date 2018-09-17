import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde
import os
# import readline         # workaround for bug related to rpy2 (https://github.com/ContinuumIO/anaconda-issues/issues/152)
import rpy2.robjects as ro
from statsmodels.nonparametric.smoothers_lowess import lowess

from . import DataLoading


def preprocess_mass_spec_file(input_filename, output_filename, organism_name, n_peaks_to_keep, peak_encoding,
                              max_intensity_value=5000,
                              max_mz_value=5000, filter_on_charge=None, include_charge_in_encoding=False,
                              include_molecular_weight_in_encoding=False, use_smoothed_intensities=False,
                              smoothing_method="loess", smoothness_sigma=1, smoothness_frac=0.3,
                              smoothing_n_gaussians=15, smoothness_spar=0.3):
    """
    preprocesses the mass spec file by fixing the number of peaks for each spectra to n_peaks_to_keep
    (n_peaks_to_keep peaks with the highest intensity are kept; rest is ignored); and creates the feature representation
    as input for the network
    :param input_filename: filename of the mass spec data
    :param output_filename: filename the preprocessed data should be written to
    :param organism_name: e.g. "yeast"; used for output filename
    :param n_peaks_to_keep: how many peaks one spectrum should contain
    :param filter_on_charge: which charge should be kept (e.g. "2" to keep only spectra with a charge of 2)
    :param peak_encoding: "location" or "distance": encoding of the peaks
        - "location":
            (1) square root of its height
            (2) its location (mz distance from 0)
            (3) its location relative to the precursor (mz distance relative to precursor)
        - "distance":
            (1) square root of its height
            (2) its location relative to the successor (mz distance relative to successor)
            (3) its location relative to the precursor (mz distance relative to precursor)
        - "binned":
            (1) m/z values binned in k bins
            (2) intensity of the respective bin
        - "only_mz":
            (1) mz distance relative to successor
    :param max_intensity_value: at which intensity value a peak should be considered as an outlier
    :param max_mz_value: at which m/z value a peak should be considered as an outlier
    :param include_charge_in_encoding: whether or not the charge should be appended to the peak_encoding
    :param include_molecular_weight_in_encoding: whether or not the molecular weight should be appended to the
    peak_encoding
    :param use_smoothed_intensities: whether or not the intensities should be smoothed
    :param smoothing_method: ["loess", "gaussian_filter", "spline"] which smoothness method to use
    :param smoothness_sigma: parameter for smoothing the intensities using the gaussian filter; the higher the smoother
    :param smoothness_frac: parameter for smoothing the intensities using loess; the higher the smoother
    :param smoothing_n_gaussians: how many gaussians should be used for smoothing the intensities
    :param smoothness_spar: parameter for smoothing the intensities using spline; the higher the smoother
    :return:
    """

    # load the input data
    print("Loading data..")

    # preprocess the smoothed intensities
    if use_smoothed_intensities:
        # check if they have been already preprocessed previously
        if not os.path.isfile(output_filename):
            preprocess_smoothed_intensities(n_peaks_to_keep, smoothing_n_gaussians, max_mz_value, max_intensity_value,
                                            output_filename=output_filename, sigma=smoothness_sigma,
                                            smoothing_method=smoothing_method, frac=smoothness_frac,
                                            spar=smoothness_spar)

    if input_filename.endswith(".mgf"):
        if use_smoothed_intensities:
            spectra = DataLoading.load_mgf_file(output_filename.replace(".txt", ".mgf"))
        else:
            spectra = DataLoading.load_mgf_file(input_filename.replace(".txt", ".mgf"))
    else:
        if use_smoothed_intensities:
            spectra = DataLoading.load_msp_file(output_filename.replace(".txt", ".msp"))
        else:
            spectra = DataLoading.load_msp_file(input_filename.replace(".txt", ".msp"))

    # get the encoding for the peaks
    print("Encoding the data..")
    peak_features = get_peaks_encoding(spectra, peak_encoding=peak_encoding,
                                       n_peaks_to_keep=n_peaks_to_keep, filter_on_charge=filter_on_charge,
                                       include_charge_in_encoding=include_charge_in_encoding,
                                       include_molecular_weight_in_encoding=include_molecular_weight_in_encoding,
                                       max_intensity_value=max_intensity_value, max_mz_value=max_mz_value)

    # filter the outliers (if we have location or distance encoding)
    print("Filtering the outliers..")
    n_data_points = len(peak_features)
    if peak_encoding == "location" or peak_encoding == "distance":
        feature_dim = n_peaks_to_keep * 3
        # filter for intensity
        peak_features = peak_features[np.all(peak_features[:, :feature_dim][:, ::3] < np.sqrt(max_intensity_value), axis=1)]
        # filter for negative values
        peak_features = peak_features[np.all(peak_features >= 0, axis=1)]
        print(str(n_data_points - len(peak_features)) + " outliers out of " + str(n_data_points) + " data points have "
                                                                                                   "been removed.")

    # save the numpy array to the file
    np.savetxt(output_filename, peak_features)

    print("Pre-processed data saved to " + output_filename)


def get_peaks_encoding(mass_spec_data, peak_encoding, n_peaks_to_keep=30, filter_on_charge=None,
                       include_charge_in_encoding=True, include_molecular_weight_in_encoding=True,
                       max_intensity_value=5000, max_mz_value=5000):
    """
    preprocesses the mass spec data by fixing the number of peaks for each spectra to n_peaks_to_keep
    (n_peaks_to_keep peaks with the highest intensity are kept; rest is ignored); and creates the feature representation
    as input for the network
    :param mass_spec_data: list of dictionaries holding the data
    :param peak_encoding: "location" or "distance": encoding of the peaks
        - "location":
            (1) square root of its height
            (2) its location (mz distance from 0)
            (3) its location relative to the precursor (mz distance relative to precursor)
        - "distance":
            (1) square root of its height
            (2) its location relative to the successor (mz distance relative to successor)
            (3) its location relative to the precursor (mz distance relative to precursor)
        - "binned":
            (1) m/z values binned in k bins
            (2) intensity of the respective bin
        - "only_mz":
            (1) mz distance relative to successor
    :param n_peaks_to_keep: how many peaks the spectra should contain
    :param filter_on_charge: which charge should be kept (e.g. "2" to keep only spectra with a charge of 2)
    :param include_charge_in_encoding: whether or not the charge should be appended to the peak_encoding
    :param include_molecular_weight_in_encoding: whether or not the molecular weight should be appended to the
    peak_encoding
    :param max_intensity_value: at which intensity value a peak should be considered as an outlier
    :param max_mz_value: at which m/z value a peak should be considered as an outlier
    :return: list of lists with the feature representation of the spectra
    """

    # get the peaks for all spectra
    peaks = [spectrum["peaks"] for spectrum in mass_spec_data]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, n_peaks_to_keep) for a in peaks]
    indices_to_keep = [i for i, e in enumerate(filtered_peaks) if e is not None]
    mass_spec_data = mass_spec_data[indices_to_keep]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 0] < max_mz_value, axis=1)        # m/z values
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]
    mass_spec_data = mass_spec_data[indices_to_keep]

    indices_to_keep = np.all(filtered_peaks[:, :, 1] < max_intensity_value, axis=1)        # intensities
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]
    mass_spec_data = mass_spec_data[indices_to_keep]

    # create the features for the peaks ((1) square root of its height (2) its location (mz distance from 0) and
    # (3) its location relative to the precursor (mz distance relative to precursor))
    peak_features = np.array([create_features_for_peak(peak, peak_encoding) for peak in filtered_peaks])

    # get the charge
    charge_list = np.array([spectrum["charge"] for spectrum in mass_spec_data]).reshape(-1, 1)

    # get the molecular weight
    molecular_weight_list = np.array([spectrum["pepmass"] for spectrum in mass_spec_data]).reshape(-1, 1)

    # we are only interested in spectra with charge "filter_on_charge"
    if filter_on_charge:
        indices = [i for i, e in enumerate(charge_list) if e == [filter_on_charge]]
        peak_features = peak_features[indices]
        charge_list = charge_list[indices]
        molecular_weight_list = molecular_weight_list[indices]

    if include_charge_in_encoding and include_molecular_weight_in_encoding \
            or peak_encoding == "only_mz_charge_label" and include_molecular_weight_in_encoding:
        # combine the peaks, the charges and the molecular weight in one numpy array
        pre_processed_mass_spec_data = np.hstack((peak_features, charge_list, molecular_weight_list)).astype(float)
    elif include_charge_in_encoding or peak_encoding == "only_mz_charge_label":
        # combine the peaks and the charges in one numpy array
        pre_processed_mass_spec_data = np.hstack((peak_features, charge_list)).astype(float)
    elif include_molecular_weight_in_encoding:
        # combine the peaks and the molecular weight in one numpy array
        pre_processed_mass_spec_data = np.hstack((peak_features, molecular_weight_list)).astype(float)
    else:
        # combine the peaks, the charges and the molecular weight in one numpy array
        pre_processed_mass_spec_data = peak_features.astype(float)

    return pre_processed_mass_spec_data


def create_features_for_peak(peak, peak_encoding):
    """
    creates the feature vector for the list of peaks and returns it as array with shape [-1, 3]:
    represents each peak by 3 numbers (1) square root of its height (2) its location (mz distance from 0) and
    (3) its location relative to the precursor (mz distance relative to precursor)
    :param peak: np array [mz_values, intensities]
    :param peak_encoding: "location" or "distance": encoding of the peaks
    - "location":
        (1) square root of its height
        (2) its location (mz distance from 0)
        (3) its location relative to the precursor (mz distance relative to precursor)
    - "distance":
        (1) square root of its height
        (2) its location relative to the successor (mz distance relative to successor)
        (3) its location relative to the precursor (mz distance relative to precursor)
    """

    # (1) square root of its height
    intensities = peak[:, 1]
    try:
        intensities = np.sqrt(intensities)
    except TypeError:
        intensities = np.sqrt(intensities.astype(float))      # convert to float if necessary

    # get the mz values
    mz_values = peak[:, 0].astype(float)
    # calculate the relativ distances between the current value and its precursor
    rel_distances = [round(x - y, 2) for x, y in zip(mz_values[1:], mz_values)]

    # (2) its location (mz distance from 0)
    if peak_encoding == "location":
        second_feature_vector = mz_values
    # (2) its location relative to the successor (mz distance relative to successor)
    elif peak_encoding == "distance":
        second_feature_vector = rel_distances.copy()
        second_feature_vector.insert(len(second_feature_vector), 0)  # last element has distance 0 to its successor
    elif peak_encoding == "raw":
        intensities = peak[:, 1]
        return np.hstack((mz_values, intensities)).reshape(-1, 2, order='F').reshape(-1)
    elif peak_encoding == "raw_intensities_sqrt":
        intensities = np.sqrt(peak[:, 1])
        return np.hstack((mz_values, intensities)).reshape(-1, 2, order='F').reshape(-1)
    elif peak_encoding == "raw_sqrt":
        mz_values = np.sqrt(peak[:, 0])
        intensities = np.sqrt(peak[:, 1])
        return np.hstack((mz_values, intensities)).reshape(-1, 2, order='F').reshape(-1)
    elif peak_encoding == "only_mz" or peak_encoding == "only_mz_charge_label":
        return mz_values
    elif peak_encoding == "only_intensities":
        return np.sqrt(peak[:, 1])
    else:
        raise ValueError(str(peak_encoding) + " is an invalid value for this parameter. Try 'location' or 'distance'.")

    # (3) its location relative to the precursor (mz distance relative to precursor)
    third_feature_vector = rel_distances.copy()
    third_feature_vector.insert(0, mz_values[0])  # first element has distance 0 to its precursor

    return np.hstack((intensities, second_feature_vector, third_feature_vector)).reshape(-1, 3, order='F').reshape(-1)


def filter_spectrum(peaks_in_spectrum, n_peaks_to_keep):
    """
    filters the spectra for the highest n_peaks_to_keep; spectra with less peaks_in_spectrum are ignored
    :param peaks_in_spectrum: list of peaks in the current spectrum holding the m/z value and the intensity
    :param n_peaks_to_keep: how many peaks_in_spectrum each spectra
    :return: filtered lists
    """

    if len(peaks_in_spectrum) < n_peaks_to_keep:
        return None

    # convert list to np.array
    peaks_in_spectrum = np.array(peaks_in_spectrum)

    # get the intensities
    intensities = peaks_in_spectrum[:, 1]

    # get the indices for the n highest intensities
    indices_to_keep = np.argsort(intensities)[:n_peaks_to_keep]

    # sort the indices, so the m/z values are in proper order
    indices_to_keep = np.sort(indices_to_keep)

    return peaks_in_spectrum[indices_to_keep]


def plot_spectras_stem_plot():

    unidentified_spectra = np.loadtxt("../../data/mass_spec_data/yeast/yeast_unidentified_distance_charge_2_n_peaks_50_max_intensity_5000.txt")
    identified_spectra = np.loadtxt("../../data/mass_spec_data/yeast/yeast_identified_distance_charge_2_n_peaks_50_max_intensity_5000.txt")

    all_spectra = np.concatenate((identified_spectra, unidentified_spectra))

    mass_spec_data_properties = {"organism_name": "yeast", "peak_encoding": "location",
                                 "include_charge_in_encoding": True, "include_molecular_weight_in_encoding": True,
                                 "charge": "2", "normalize_data": False, "n_peaks_to_keep": 50,
                                 "max_intensity_value": 5000}

    from util.VisualizationUtils import reconstruct_spectrum_from_feature_vector

    mz_values_all, intensities_all, _, _ \
        = reconstruct_spectrum_from_feature_vector(all_spectra, 152, mass_spec_data_properties)
    mean_mz_values_all = np.mean(mz_values_all, axis=0)
    mean_intensities_all = np.mean(intensities_all, axis=0)

    mz_values_unidentified, intensities_unidentified, _, _ \
        = reconstruct_spectrum_from_feature_vector(unidentified_spectra, 152, mass_spec_data_properties)
    mean_mz_values_unidentified = np.mean(mz_values_unidentified, axis=0)
    mean_intensities_unidentified = np.mean(intensities_unidentified, axis=0)

    mz_values_identified, intensities_identified, _, _ \
        = reconstruct_spectrum_from_feature_vector(identified_spectra, 152, mass_spec_data_properties)
    mean_mz_values_identified = np.mean(mz_values_identified, axis=0)
    mean_intensities_identified = np.mean(intensities_identified, axis=0)

    plt.stem(mean_mz_values_all, mean_intensities_all, 'b', label="all", markerfmt=' ')
    plt.stem(mean_mz_values_unidentified, mean_intensities_unidentified, 'r', label="unidentified", markerfmt=' ')
    plt.stem(mean_mz_values_identified, mean_intensities_identified, 'g', label="identified", markerfmt=' ')

    plt.legend()
    plt.ylabel("intensity")
    plt.xlabel("m/z value")
    plt.title("Avg. spectra - yeast")

    plt.show()


def plot_spectras_boxplot():
    unidentified_spectra = np.loadtxt("../../data/mass_spec_data/yeast/yeast_unidentified_distance_charge_2_n_peaks_50_max_intensity_5000.txt")
    identified_spectra = np.loadtxt("../../data/mass_spec_data/yeast/yeast_identified_distance_charge_2_n_peaks_50_max_intensity_5000.txt")

    all_spectra = np.concatenate((identified_spectra, unidentified_spectra))

    mass_spec_data_properties = {"organism_name": "yeast", "peak_encoding": "location",
                                 "include_charge_in_encoding": True, "include_molecular_weight_in_encoding": True,
                                 "charge": "2", "normalize_data": False, "n_peaks_to_keep": 50,
                                 "max_intensity_value": 5000}

    from util.VisualizationUtils import reconstruct_spectrum_from_feature_vector

    mz_values_all, intensities_all, _, _ \
        = reconstruct_spectrum_from_feature_vector(all_spectra, 152, mass_spec_data_properties)
    mean_mz_values_all = np.mean(mz_values_all, axis=0)
    mean_intensities_all = np.mean(intensities_all, axis=0)

    mz_values_unidentified, intensities_unidentified, _, _ \
        = reconstruct_spectrum_from_feature_vector(unidentified_spectra, 152, mass_spec_data_properties)
    mean_mz_values_unidentified = np.mean(mz_values_unidentified, axis=0)
    mean_intensities_unidentified = np.mean(intensities_unidentified, axis=0)

    mz_values_identified, intensities_identified, _, _ \
        = reconstruct_spectrum_from_feature_vector(identified_spectra, 152, mass_spec_data_properties)
    mean_mz_values_identified = np.mean(mz_values_identified, axis=0)
    mean_intensities_identified = np.mean(intensities_identified, axis=0)

    values_to_plot = [mean_mz_values_all, mean_intensities_all,
                      mean_mz_values_unidentified, mean_intensities_unidentified,
                      mean_mz_values_identified, mean_intensities_identified]

    plt.boxplot(values_to_plot, labels=["mean_mz_values_all", "mean_intensities_all",
                                        "mean_mz_values_unidentified", "mean_intensities_unidentified",
                                        "mean_mz_values_identified", "mean_intensities_identified"])

    plt.legend()
    plt.title("Avg. spectra - yeast")
    plt.show()

    return


def create_binned_encoding(spectra, n_bins, return_type=float):
    """
    creates the binned encoding for the mass spec data, where the m/z values are put into n_bins bins with the
    respective intensity for the bin
    :param spectra: array of dictionaries holding the spectra
    :param n_bins: number of bins for the m/z values
    :param return_type: return type of the array
    :return:
    """

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    print("without filtering: {}".format(filtered_peaks.shape[0]))

    # filter outliers for intensity
    indices_to_keep = np.all(filtered_peaks[:, :, 1] < 2500, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]

    print("after filtering: {}".format(filtered_peaks.shape[0]))

    # get the m/z values and the intensities
    mz_values = filtered_peaks[:, :, 0]
    intensities = filtered_peaks[:, :, 1]

    # create the bins
    bin_size = 2500 / n_bins
    bins = [i * bin_size for i in range(0, n_bins + 1)]

    # bin the mz values
    mz_values_binned = np.array([np.histogram(mz_value, bins)[0] for mz_value in mz_values])

    # map the intensity values to the mz bins
    def map_intensity_to_mz_bins(mz_array, intensity_array):
        counter = 0
        for index_with_value in np.where(mz_array)[0]:
            bin_value = int(mz_array[index_with_value])
            mz_array[index_with_value] = np.sum(intensity_array[counter:counter + bin_value])
            counter += bin_value
        return mz_array

    final_data = np.array([map_intensity_to_mz_bins(mz_value, intensity_value) for mz_value, intensity_value in
                           zip(mz_values_binned.astype(return_type), intensities)])

    return final_data


def test_binning_of_mass_spec_data():

    # TODO: mass spec properties file

    unidentified_spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")

    identified_spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")

    unidentified_spectra_pre_processed = create_binned_encoding(unidentified_spectra, 1000, float)
    identified_spectra_pre_processed = create_binned_encoding(identified_spectra, 1000, float)

    np.savetxt("../../data/mass_spec_data/yeast/yeast_unidentified_binned.txt", unidentified_spectra_pre_processed)
    np.savetxt("../../data/mass_spec_data/yeast/yeast_identified_binned.txt", identified_spectra_pre_processed)


def test_restore_binned_data():

    mass_spec_data_properties = {"organism_name": "yeast", "peak_encoding": "binned",
                                 "include_charge_in_encoding": False, "include_molecular_weight_in_encoding": True,
                                 "charge": "2", "normalize_data": True, "n_peaks_to_keep": 50,
                                 "max_intensity_value": 5000}

    data = DataLoading.get_input_data("mass_spec", filepath="../../data", color_scale="gray_scale", data_normalized=False,
                          add_noise=False, mass_spec_data_properties=mass_spec_data_properties)

    spectra, _ = data.train.next_batch(10)

    # spectra = np.loadtxt("../../data/mass_spec_data/yeast/yeast_unidentified_binned.txt")

    print(spectra.shape)

    bin_size = 2500 / spectra.shape[1]

    print(bin_size)

    print(spectra[0, :].shape)

    def keep_top_peaks(spectrum, n_peaks_to_keep=50):
        indices_to_keep = np.argsort(spectrum)[::-1][:n_peaks_to_keep]
        # sort the indices, so the m/z values are in proper order
        indices_to_keep = np.sort(indices_to_keep)
        return indices_to_keep, spectrum[indices_to_keep]

    la = np.array([keep_top_peaks(spectrum) for spectrum in spectra])

    mz_values = la[:, 0, :]*bin_size
    intensities = la[:, 1, :]

    print(intensities.shape)
    print(intensities[0])
    print(intensities[2])

    print(mz_values[0])
    print(mz_values[0]*bin_size)

    plt.stem(mz_values[0], intensities[0])
    plt.show()


def preprocess_only_mz():
    """
    preprocess the mass spec data to keep only the distances between the m/z values
    :return:
    """

    is_data_unidentified = False

    if is_data_unidentified:
        spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")
    else:
        spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    print(filtered_peaks.shape)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 0] < 4000, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]

    # we are only interested in the m/z values
    mz_values = filtered_peaks[:, :, 0]

    print(filtered_peaks.shape)

    # add a zero to calculate the distances between the data points
    mz_values = np.append(np.zeros((mz_values.shape[0], 1)), mz_values, axis=1)

    # calculate the relativ distances between the current value and its precursor
    rel_distances = np.array([[round(x - y, 2) for x, y in zip(mz_values_entry[1:], mz_values_entry)] for mz_values_entry in mz_values])

    print(rel_distances.shape)
    print(mz_values.shape)

    print(rel_distances[0])
    print(mz_values[0])
    # test reconstruction
    reconstructed = np.array([[sum(entry[:i + 1]) for i, x in enumerate(entry)] for entry in rel_distances])
    print(reconstructed.shape)
    print(reconstructed[0])

    print(np.max(mz_values))

    if is_data_unidentified:
        np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_mz.txt', rel_distances)
    else:
        np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_mz.txt', rel_distances)


def preprocess_only_intensities():
    """
    preprocess the mass spec data to keep only the distances between the m/z values
    :return:
    """

    is_data_unidentified = False
    use_log = True

    if is_data_unidentified:
        spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")
    else:
        spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    print(filtered_peaks.shape)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 1] < 5000, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]

    # we are only interested in the intensities
    intensities = filtered_peaks[:, :, 1]

    print(np.max(intensities))
    print(np.min(intensities))

    if use_log:
        intensities = np.sqrt(intensities)

    # plt.boxplot(intensities)
    # plt.ylabel("Intensity")
    # plt.xlabel("Peaks")
    # plt.show()

    print(intensities.shape)
    print(np.max(intensities))
    print(np.min(intensities))

    if is_data_unidentified:
        if use_log:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_intensities_sqrt.txt', intensities)
        else:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_intensities.txt', intensities)
    else:
        if use_log:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_intensities_sqrt.txt', intensities)
        else:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_intensities.txt', intensities)


def preprocess_only_intensities_distance_encoding():
    """
    preprocess the mass spec data to keep only the distances between the intensities
    :return:
    """

    is_data_unidentified = True
    use_log = False

    if is_data_unidentified:
        spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")
    else:
        spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    print(filtered_peaks.shape)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 1] < 5000, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]

    # we are only interested in the intensities
    intensities = filtered_peaks[:, :, 1]

    print(np.max(intensities))
    print(np.min(intensities))

    # add a zero to calculate the distances between the data points
    intensities = np.append(np.zeros((intensities.shape[0], 1)), intensities, axis=1)

    # calculate the relativ distances between the current value and its precursor
    rel_distances = np.array([[round(x - y, 2) for x, y in zip(intensity_entry[1:], intensity_entry)] for intensity_entry in intensities])

    print(rel_distances[0])
    print(intensities[0])
    # test reconstruction
    reconstructed = np.array([[sum(entry[:i + 1]) for i, x in enumerate(entry)] for entry in rel_distances])
    print(reconstructed.shape)
    print(reconstructed[0])

    if np.allclose(intensities[0][1:], reconstructed[0], atol=0.001):
        print("Reconstruction worked!")
    else:
        raise ValueError("Reconstruction failed!")

    if use_log:
        rel_distances = np.sqrt(rel_distances)

    # plt.boxplot(intensities)
    # plt.ylabel("Intensity")
    # plt.xlabel("Peaks")
    # plt.show()

    print(intensities.shape)
    print(np.max(intensities))
    print(np.min(intensities))

    if is_data_unidentified:
        if use_log:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_intensities_sqrt_distance_encoding.txt', rel_distances)
        else:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_intensities_distance_encoding.txt', rel_distances)
    else:
        if use_log:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_intensities_sqrt_distance_encoding.txt', rel_distances)
        else:
            np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_intensities_distance_encoding.txt', rel_distances)


def density_scatter_plot():
    unidentified_spectra = DataLoading.load_mgf_file(
        "../../data/mass_spec_data/yeast/yeast_unidentified.mgf")

    peaks = [spectrum["peaks"] for spectrum in unidentified_spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None])

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 1] < 2500, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]

    # reduce data set size
    filtered_peaks = filtered_peaks[:500]

    mz_values = filtered_peaks[:, :, 0]
    intensities = filtered_peaks[:, :, 1]

    # flatten array
    mz_values = mz_values.flatten()
    intensities = intensities.flatten()

    xy = np.vstack([mz_values, intensities])

    print(xy.shape)

    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(mz_values, intensities, c=z, s=100, edgecolor='')
    plt.show()


def preprocess_only_mz_with_charge_as_label(is_data_unidentified=True):
    """
    preprocess the mass spec data to keep only the distances between the m/z values and the charges as label
    :return:
    """

    if is_data_unidentified:
        spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")
    else:
        spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = [filter_spectrum(a, 50) for a in peaks]
    indices_to_keep = [i for i, e in enumerate(filtered_peaks) if e is not None]
    charge_list = np.array([spectrum["charge"] for spectrum in spectra])[indices_to_keep].reshape(-1, 1).astype(int)

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    print(filtered_peaks.shape)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 0] < 4000, axis=1)
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]
    charge_list = charge_list[indices_to_keep]

    # we are only interested in the m/z values
    mz_values = filtered_peaks[:, :, 0]

    print(filtered_peaks.shape)

    # add a zero to calculate the distances between the data points
    mz_values = np.append(np.zeros((mz_values.shape[0], 1)), mz_values, axis=1)

    # calculate the relativ distances between the current value and its precursor
    rel_distances = np.array(
        [[round(x - y, 2) for x, y in zip(mz_values_entry[1:], mz_values_entry)] for mz_values_entry in mz_values])

    print(rel_distances.shape)
    print(mz_values.shape)

    print(rel_distances[0])
    print(mz_values[0])
    # test reconstruction
    reconstructed = np.array([[sum(entry[:i + 1]) for i, x in enumerate(entry)] for entry in rel_distances])
    print(reconstructed.shape)
    print(reconstructed[0])

    print(np.max(mz_values))

    print(charge_list.shape)
    print(rel_distances.shape)
    print(np.array([spectrum["charge"] for spectrum in spectra]).reshape(-1, 1).shape)

    rel_distances_and_charge = np.hstack((rel_distances, charge_list))

    if is_data_unidentified:
        np.savetxt('../../data/mass_spec_data/yeast/yeast_unidentified_only_mz_charge_label.txt', rel_distances_and_charge)
    else:
        np.savetxt('../../data/mass_spec_data/yeast/yeast_identified_only_mz_charge_label.txt', rel_distances_and_charge)


def smooth_spectrum_with_splines(mz_values, intensities, spar):
    """
    smooths a single spectrum with the spline smoothing; see
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/smooth.spline.html
    :param mz_values: list of m/z values of the spectrum
    :param intensities: list of intensities of the spectrum
    :param spar: smoothing factor
    :return: np.array of the smoothed intensities
    """

    # convert the lists into r vectors
    mz_values_r_vector = ro.FloatVector(mz_values)
    intensities_r_vector = ro.FloatVector(intensities)

    # smooth the intensities
    splines_smoothing_function = ro.r["smooth.spline"]
    smoothed_intensities_splines = splines_smoothing_function(mz_values_r_vector, intensities_r_vector, spar=spar,
                                                      **{"all.knots": True}).rx2('y')
    smoothed_intensities_splines = np.asarray(smoothed_intensities_splines)

    return smoothed_intensities_splines


def smooth_spectrum_with_loess(mz_values, intensities, span):
    """
    smooths a single spectrum with the spline smoothing; see
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html
    :param mz_values: list of m/z values of the spectrum
    :param intensities: list of intensities of the spectrum
    :param span: smoothing factor
    :return: np.array of the smoothed intensities
    """

    # convert the lists into r vectors
    mz_values_r_vector = ro.FloatVector(mz_values)
    intensities_r_vector = ro.FloatVector(intensities)

    df = {"MZ": mz_values_r_vector, "Int": intensities_r_vector}
    spectrum_dataframe = ro.DataFrame(df)

    # smooth the intensities
    loess_smoothing_function = ro.r["loess"]
    predict_function = ro.r["predict"]
    smoothed_dataframe = predict_function(loess_smoothing_function('Int ~ MZ', spectrum_dataframe, span=span))
    smoothed_intensities = np.asarray(smoothed_dataframe)

    return smoothed_intensities


def preprocess_smoothed_intensities(n_peaks_to_keep, n_gaussians, max_mz_value, max_intensity_value, output_filename,
                                    smoothing_method, sigma=1, frac=0.3, spar=0.3):
    """
    smoothes the intensities by using a number of gaussian evenly distributed along the m/z values [0, max_mz_value]
    using the scipy function gaussian_filter with the respective sigma
    :param n_peaks_to_keep: how many peaks one spectrum should contain
    :param n_gaussians: how many gaussians should be used for smoothing the intensities
    :param max_mz_value: at which value a peak should be considered as an outlier and thus be ignored
    :param max_intensity_value: at which value a peak should be considered as an outlier and thus be ignored
    :param output_filename: filename to write
    :param smoothing_method: ["gaussian_filter", "loess", "spline"]; which method should be used for smoothing
    :param sigma: scalar or sequence of scalars; Standard deviation for Gaussian kernel. The standard deviations of
    the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for
    all axes.
    :param frac: Between 0 and 1. smoothness factor for loess smoothing
    :param spar: Between 0 and 1. smoothness factor for spline smoothing
    :return:
    """

    print("Smoothing the intensities..")

    is_data_identified = "unidentified" not in output_filename
    if is_data_identified:
        spectra = DataLoading.load_msp_file("../../data/mass_spec_data/yeast/yeast_identified.msp")
        output_filename = output_filename.replace(".txt", ".msp")
    else:
        spectra = DataLoading.load_mgf_file("../../data/mass_spec_data/yeast/yeast_unidentified.mgf")
        output_filename = output_filename.replace(".txt", ".mgf")

    peaks = [spectrum["peaks"] for spectrum in spectra]

    # filter to keep only the n highest peaks
    filtered_peaks = np.array([filter_spectrum(a, n_peaks_to_keep=n_peaks_to_keep) for a in peaks])
    # get the indices where data is not None
    valid_indices = [j for j, i in enumerate(filtered_peaks) if i is not None]
    spectra = spectra[valid_indices]

    # remove None from the list
    filtered_peaks = np.array([i for i in filtered_peaks if i is not None]).astype(float)

    # filter outliers
    indices_to_keep = np.all(filtered_peaks[:, :, 0] < max_mz_value, axis=1)        # m/z values
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]
    spectra = spectra[indices_to_keep]

    indices_to_keep = np.all(filtered_peaks[:, :, 1] < max_intensity_value, axis=1)        # intensities
    filtered_peaks = filtered_peaks[indices_to_keep, :, :]
    spectra = spectra[indices_to_keep]

    intensities = np.array(filtered_peaks[:, :, 1])
    mz_values = np.array(filtered_peaks[:, :, 0])

    """
    smooth the intensities
    """
    smoothed_intensities = None
    if smoothing_method == "gaussian_filter":
        # smooth using a 1d gaussian filter
        smoothed_intensities = gaussian_filter1d(intensities, sigma=sigma)

    elif smoothing_method == "loess":
        # smooth using loess
        smoothed_intensities = np.array([smooth_spectrum_with_loess(mz_values[i], intensities[i], span=frac)
                                         for i in range(mz_values.shape[0])])

    elif smoothing_method == "lowess":
        # smooth using lowess
        smoothed_intensities = np.array([lowess(intensities[i], mz_values[i], is_sorted=True, return_sorted=False,
                                                frac=frac, it=0) for i in range(mz_values.shape[0])])

    elif smoothing_method == "spline":
        # smooth using splines
        smoothed_intensities = np.array([smooth_spectrum_with_splines(mz_values[i], intensities[i], spar=spar)
                                         for i in range(mz_values.shape[0])])

    else:
        raise ValueError("Smoothing method " + smoothing_method + " is invalid")

    # remove nan
    not_nan_indices = np.all(np.logical_not(np.isnan(smoothed_intensities)), axis=1)
    smoothed_intensities = smoothed_intensities[not_nan_indices]
    spectra = spectra[not_nan_indices]

    # remove negative data points
    indices_to_keep = np.all(0 < smoothed_intensities, axis=1)
    spectra = spectra[indices_to_keep]
    smoothed_intensities = smoothed_intensities[indices_to_keep]

    smoothed_intensities = np.around(smoothed_intensities, decimals=3)

    """
    write the smoothed intensities to some file
    """
    if is_data_identified:
        write_smoothed_intensities_to_msp_file(smoothed_intensities, spectra, filename=output_filename)
    else:
        write_smoothed_intensities_to_mgf_file(smoothed_intensities, spectra, filename=output_filename)


def write_smoothed_intensities_to_msp_file(smoothed_intensities, spectra, filename):
    """
    writes the smoothed intensities back to the respective .msp file
    :param smoothed_intensities: [n_datapoints, n_peaks] array holding the smoothed intensities
    :param spectra: [n_datapoints, n_peaks] array of dictionaries holding the title, charge, pepmass and m/z values
    :param filename: filename of the file to write
    :return:
    """
    num_peaks = smoothed_intensities.shape[1]
    with open(filename, 'w') as output_file:
        for spectrum, spectrum_smoothed_intensities in zip(spectra, smoothed_intensities):
            output_file.write("Name: " + spectrum["title"] + "\n")
            output_file.write("MW: " + spectrum["pepmass"] + "\n")
            output_file.write("Comment: " + spectrum["comment"] + "\n")
            output_file.write("Num peaks: " + str(num_peaks) + "\n")
            for peak, peak_smoothed_intensity in zip(spectrum["peaks"], spectrum_smoothed_intensities):
                output_file.write(str(peak[0]) + " " + str(peak_smoothed_intensity) + "\n")
            output_file.write("\n")


def write_smoothed_intensities_to_mgf_file(smoothed_intensities, spectra, filename):
    """
    writes the smoothed intensities back to the respective .mgf file
    :param smoothed_intensities: [n_datapoints, n_peaks] array holding the smoothed intensities
    :param spectra: [n_datapoints, n_peaks] array of dictionaries holding the title, charge, pepmass and m/z values
    :param filename: filename of the file to write
    :return:
    """
    with open(filename, 'w') as output_file:
        for spectrum, spectrum_smoothed_intensities in zip(spectra, smoothed_intensities):
            output_file.write("BEGIN IONS\n")
            output_file.write("TITLE=" + spectrum["title"] + ",sequence=" + spectrum["sequence"] + "\n")
            output_file.write("PEPMASS=" + spectrum["pepmass"] + "\n")
            output_file.write("CHARGE=" + spectrum["charge"] + "+\n")
            output_file.write("SEQUENCE=" + spectrum["sequence"] + "\n")
            for peak, peak_smoothed_intensity in zip(spectrum["peaks"], spectrum_smoothed_intensities):
                output_file.write(str(peak[0]) + " " + str(peak_smoothed_intensity) + "\n")
            output_file.write("END IONS\n\n")

