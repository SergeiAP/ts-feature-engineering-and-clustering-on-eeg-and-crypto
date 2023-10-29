from typing import Iterable
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.fftpack import fft

from feature_preproc.detect_peaks import detect_peaks


def get_values(y_values: Iterable,
               T: float,
               N: int,
               f_s: float,
               sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Get values itself and frequencies"""
    y_values = y_values
    x_values = np.array([sample_rate * kk for kk in range(0, len(y_values))])
    return x_values, y_values


def get_fft_values(y_values: Iterable,
                   T: float,
                   N: int,
                   f_s: float,
                   sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Get Fourier coefficients and frequencies"""
    # since we are only interested in the magnitude of the amplitudes, 
    # we use np.abs() to take the real part of the frequency spectrum.
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


def get_psd_values(y_values: Iterable,
                   T: float,
                   N: int,
                   f_s: float,
                   sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Get Power Spectral Density and frequencies"""
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, np.abs(psd_values)

def get_first_n_peaks(x, y, no_peaks: int=5) -> tuple[list, list]:
    """Return first n peaks from """
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
    
def get_features(x_values: Iterable,
                 y_values: Iterable,
                 mph: float,
                 no_peaks: int):
    """_summary_

    Args:
        x_values (Iterable): _description_
        y_values (Iterable): _description_
        mph (float): _description_
        no_peaks (int): _description_

    Returns:
        _type_: _description_
    """
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], 
                                         y_values[indices_peaks], 
                                         no_peaks)
    return peaks_x + peaks_y


def extract_features_labels(x: pd.DataFrame,
                            y: pd.Series,
                            T: float,
                            N: int,
                            f_s: float,
                            no_peaks: int,
                            denominator: float,
                            percentile: int,
                            sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Make Fourier and PSD transformation, then take """
    list_of_features = []
    list_of_labels = []
    for signal_no in tqdm(range(0, len(x)), leave=False):
        list_of_labels.append(y[signal_no])
        signal = x.iloc[signal_no, :]
        
        signal_min = np.nanpercentile(signal, percentile)
        signal_max = np.nanpercentile(signal, 100-percentile)
        #ijk = (100 - 2*percentile)/10
        mph = signal_min + (signal_max - signal_min) / denominator
        
        features = get_features(*get_psd_values(signal, T, N, f_s, sample_rate), 
                                mph, no_peaks)
        features = get_features(*get_fft_values(signal, T, N, f_s, sample_rate), 
                                mph, no_peaks)
        
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)
