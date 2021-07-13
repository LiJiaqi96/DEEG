import scipy
import numpy as np

def power_spectral_density(signal, signal_frequency, window_size):
    """
    Calculate the power spectral density on EEG signal, and then extract the band power of Delta, Theta, Alpha, Beta, and Gamma band.
    We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate PSD on dim "e" and return a (p,m) numpy array.
    
    input:
        signal: numpy array. The temporal signal for feature extraction. 1d or with multiple dimensions
        signal_frequency: int. Original frequency of the signal
        window_size: int. Size of the window 
    return
        band power: tuple or numpy array, depends on the input shape
    """
    def cal_psd(s, sf, ws):
        freqs, psd = scipy.signal.welch(s, sf, nperseg=ws)
        freq_res = freqs[1] - freqs[0]

        delta = scipy.integrate.simps(psd[np.logical_and(freqs >= 0, freqs < 4)], dx=freq_res)
        theta = scipy.integrate.simps(psd[np.logical_and(freqs >= 4, freqs <= 7)], dx=freq_res)
        alpha = scipy.integrate.simps(psd[np.logical_and(freqs >= 8, freqs <= 15)], dx=freq_res)
        beta = scipy.integrate.simps(psd[np.logical_and(freqs >= 16, freqs <= 31)], dx=freq_res)
        gamma = scipy.integrate.simps(psd[np.logical_and(freqs >= 32, freqs <= freqs[-1])], dx=freq_res)
        return delta, theta, alpha, beta, gamma
    
    if len(signal.shape) == 1:
        return cal_psd(signal, signal_frequency, window_size)
    else:
        return np.apply_along_axis(cal_psd, axis=-1, arr=signal, sf=signal_frequency, ws=window_size)
