import numpy as np


def check_nan(array):
    """
    Check whether the input signal contains NaN
    input:
        array: numpy array. The temporal signal with 1d or with multiple dimensions
    return:
        None or tuple. If there is no NaN in signal, return None, else return the index where NaN occurs
    """
    if not np.isnan(array).any():
        return None
    else:
        # return format: tuple(array([a,b,c,...]), array([j,k,l,...]), ...)
        # Each element in tuple indicates the index list
        return np.where(np.isnan(array))