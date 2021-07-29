import numpy as np


def differential_entropy(signal):
    """
    Calculate the feature "Differential Entropy" following the paper: "Differential Entropy Feature for EEG-based Vigilance Estimation", with hypothesis that processed signal is normal-distributed
    We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate DE on dim "e" and return a (p,m) numpy array.
    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
    return:
        de: int or numpy array, depends on the input shape
    """
    def cal_de(s):
        return (1/2)*np.log(2*np.pi*np.e*np.std(s)**2)

    if len(array.shape)<2:
        return cal_de(array)
    else:
        return np.apply_along_axis(cal_de, axis=-1, arr=array)
