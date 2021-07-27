import numpy as np

def add_gaussian_noise(signal, snr):
    """
        Noise adding method for signal enhancement.
        input:
            array: numpy array. The input temporal signal. 1d or with multiple dimensions
            snr: parameters for size of noise
        return:
            Noise-added numpy array.
    """
    def wgn(x):
        Ps = np.sum(abs(x) ** 2) / len(x)
        Pn = Ps / (10 ** ((snr / 10)))
        noise = np.random.randn(len(x)) * np.sqrt(Pn)
        signal_add_noise = x + noise
        return signal_add_noise

    if len(signal.shape) < 2:
        return wgn(signal)
    else:
        return np.apply_along_axis(wgn, axis=-1, arr=signal)