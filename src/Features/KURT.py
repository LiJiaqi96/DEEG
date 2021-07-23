import numpy as np

def kurtosis(signal):
    """
    Calculate the feature "kurtosis" of the EEG signal
    We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate KURT on dim "e" and return a (p,m) numpy array.
    
    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
    return
        KURT: int or numpy array, depends on the input shape
    """
    def cal_kurtosis(s):
        n = len(s)

        ave1 = 0.0 
        ave2 = 0.0 
        ave4 = 0.0
        for x in s:
            ave1 += x
            ave2 += x ** 2
            ave4 += x ** 4
        ave1 /= n  
        ave2 /= n
        ave4 /= n

        sigma = np.sqrt(ave2 - ave1 ** 2)
        return ave4 / (sigma ** 4)
    
    if len(signal.shape) == 1: 
        return cal_kurtosis(signal)
    else:
        return np.apply_along_axis(cal_kurtosis, axis = -1, arr = signal)
