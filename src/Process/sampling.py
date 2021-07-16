import numpy as np


def sampling(array, interval=1, offset=0):
    """
    Down-sample the input signal with certain interval.
    input:
        array: numpy array. The input temporal signal. 1d or with multiple dimensions
        interval: int. The interval to sample EEG signal. Default is 1, which means NO down-sampling is applied
        offset: int. Sampling starts from "offset-th" data point
    return:
        sampled_array: numpy array. Down-sampled signal
    """
    if len(array.shape) < 2:
        return array[offset::interval]
    else:
        return array[:,:,offset::interval]