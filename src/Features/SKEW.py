import numpy as np

def skewness(signal):
    """
    Calculate the feature "skewness" of the EEG signal
    We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate SKEW on dim "e" and return a (p,m) numpy array.
    
    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
    return
        SKEW: int or numpy array, depends on the input shape
    """
    def cal_skewness(s):
        n = len(s)

        ave1 = 0.0 
        ave2 = 0.0 
        ave3 = 0.0 
        for x in s:
            ave1 += x
            ave2 += x**2
            ave3 += x**3
        ave1 /= n  
        ave2 /= n
        ave3 /= n

        sigma = np.sqrt(ave2 - ave1 ** 2)
        return (ave3 - 3 * ave1 * sigma ** 2 - ave1 ** 3) / (sigma ** 3)
    
    if len(signal.shape) == 1: 
        return cal_skewness(signal)
    else:
        return np.apply_along_axis(cal_skewness, axis = -1, arr = signal)
