import numpy as np
from scipy import signal


def filter(array, name="delta", order=5):
    """
    A band-pass filter to suppress signal out of frequency band
    input:
        array: numpy array. The input temporal signal with 1d
        name: str. Specify the filter type commonly used in EEG analysis. (delta, theta, alpha, beta, gamma)
        order: int. The order of Butterworth filter. Default is 5
    return:
        ts: numpy array. Filtered signal in temporal domain
    """
    if name == "delta":
        band = [1,3]
    elif name == "theta":
        band = [4,7]
    elif name == "alpha":
        band = [8,13]
    elif name == "beta":
        band = [14,30]
    elif name == "gamma":
        band = [31,50]
    else:
        raise(Exception("Invalid filter name!"))
    sos = signal.butter(order, band, 'bp', fs=1000, output='sos')
    ts = signal.sosfilt(sos, array)

    return ts


def band(array):
    """
    Filter temporal signal by all the 5 filters commonly used in EEG analysis
    input:
        signal: numpy array. The input temporal signal. 1d or with multiple dimensions
    return
        ts_dict: dictionary. Keys are filter name and values are filtered signal in temporal domain.
    """
    ts_dict = {}
    if len(array.shape) < 2:
        for name in ["delta","theta","alpha","beta","gamma"]:
            ts_dict[name] = filter(array)
    else:
        for name in ["delta","theta","alpha","beta","gamma"]:
            ts_dict[name] = np.apply_along_axis(filter, -1, array, name)
    return ts_dict
