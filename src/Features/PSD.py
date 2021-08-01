import scipy
import numpy as np

def power_spectral_density(signal, signal_frequency, window_size):
    """
    Calculate the power spectral density of given signal.
    We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate PSD on dim "e" and return a (p,m) numpy array.
    
    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
        signal_frequency: int. Original frequency of the signal
        window_size: int. Size of the window 
    return
        PSD: tuple or numpy array, depends on the input shape
    """
    def cal_psd(s, sf, ws):
        freqs, psd= scipy.signal.welch(s, fs=sf, window='hanning', nperseg=ws, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
        return psd
    
    if len(signal.shape) == 1:
        return cal_psd(signal, signal_frequency, window_size)
    else:
        return np.apply_along_axis(cal_psd, axis = -1, arr = signal, sf = signal_frequency, ws = window_size)
