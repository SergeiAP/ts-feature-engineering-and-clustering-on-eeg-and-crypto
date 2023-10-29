import numpy as np
import pandas as pd

from scipy.stats import entropy
import pywt

from collections import Counter
from tqdm.autonotebook import tqdm



def calculate_entropy(list_values) -> float:
    """(Shannon) Entropy values; entropy values can be taken as a measure of complexity
    of the signal."""
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    ent_val = entropy(probabilities)
    return ent_val

def calculate_statistics(list_values) -> list[float]:
    """Calculate and return basic statistics w/o nan's"""
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values) -> list[int]:
    """
    Calculate and return:
    * Zero crossing rate, i.e. the number of times a signal crosses y = 0
    * Mean crossing rate, i.e. the number of times a signal crosses y = mean(y)
    """
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(
        np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    """_summary_

    Args:
        list_values (_type_): _description_

    Returns:
        _type_: _description_
    """
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def get_uci_har_features(x_values: pd.DataFrame,
                         waveletname: str = 'gaus') -> np.ndarray:
    """_summary_

    Args:
        x_values (pd.DataFrame): _description_
        waveletname (str, optional): _description_. Defaults to 'gaus'.

    Returns:
        np.ndarray: _description_
    """
    uci_har_features = []
    for signal_no in tqdm(range(0, len(x_values)), leave=False):
        features = []
        signal = x_values.iloc[signal_no, :]
        # num of wavelets are automatically calculated insdie the function
        list_coeff = pywt.wavedec(signal, waveletname)
        for coeff in list_coeff:
            features += get_features(coeff)
        uci_har_features.append(features)
    return np.array(uci_har_features)
