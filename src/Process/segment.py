import numpy as np


def segment(array, length=10, overlap=1):
    """
    Generate the signal segments to augment sample size.
    input:
        array: numpy array. The input temporal signal. 1d or with multiple dimensions
        length: int. The length of segments. Default is 10, which means each signal segment contains 10 data points.
        overlap: int. The overlap length between neighboring segments. Default is 1.
    return:
        segmented_array: numpy array with shape (p,m,s,e) or (s,e), depending on the input. s is the number of segments.
    """
    if len(array.shape) < 2:
        segmented_array = [array[i:i+length] for i in range(0, len(array), length-overlap)]
        if len(segmented_array[-1]) != length:
            segmented_array = segmented_array[:-1]
        return np.array(segmented_array)
    else:
        segmented_array = [array[:,:,i:i+length] for i in range(0, array.shape[-1], length-overlap)]
        if segmented_array[-1].shape[-1] != length:
            segmented_array = segmented_array[:-1]
        return np.array(segmented_array).transpose(1,2,0,3)   # transpose numpy array to make (p,m,s,e) format