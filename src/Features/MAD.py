import numpy as np

def mean_absolute_deviation(signal):
    """
    Calculate the mean_absolute_deviation on EEG data

    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
    return
        de: int or numpy array, depends on the input shape
    """
    def cal_mad(s):
        return np.mean(np.abs(s - np.mean(s)))

    if len(signal.shape) < 2:
        return cal_mad(signal)
    else:
        return np.apply_along_axis(cal_mad, axis=-1, arr=signal)
