import numpy as np

def savgol(signal, window_size=5, polyorder=2):
    """
    Apply the Savitzky-Golay fiklter to smooth signals.
    We support both 1d array input and DEEG standard input (p,m,e).
    For the later format, we calculate DE on dim "e" and return a (p,m) numpy array.
    
    Input
    -------
    signal: numpy array or list
    window_size: int
        the length of the filter window (i.e., the number of coefficients).
        window_length` must be a positive odd integer.
        default is 5.       
    polyorder : int
        the order of the polynomial used to fit the samples.
        polyorder must be less than window_length.
        default is 2.
        
    Return
    -------
    y :ndarray
        same shape as x.
        The filtered data.
    """
    def create_x(size, rank):
    #creat weighting coefficients
        x = []
        for i in range(2 * size + 1):
            m = i - size
            row = [m**j for j in range(rank)]
            x.append(row) 
        x = np.mat(x)
        return x
    
    def SG(signal, window_size, polyorder):
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        if polyorder >= window_size:
            raise ValueError("polyorder must be less than window_lengthd")
        
        m = int((window_size - 1) / 2) 
        odata = np.array(signal).tolist()    
        for i in range(m):
            odata.insert(0, odata[0])
            odata.insert(len(odata), odata[len(odata)-1])

        # creat X matrix
        x = create_x(m, polyorder)

        # calculate the weighting coefficients
        b = (x * (x.T * x).I) * x.T
        a0 = b[m]
        a0 = a0.T

        # calculate the filtered signal
        filtered_signal = []
        for i in range(len(signal)):
            y = [odata[i + j] for j in range(window_size)]
            y1 = np.mat(y) * a0
            y1 = float(y1)
            filtered_signal.append(y1)
        return np.array(filtered_signal)

    if len(np.array(signal).shape) == 1:
        return SG(signal, window_size, polyorder)
    else:
        return np.apply_along_axis(SG, axis=-1, arr=signal, window_size=window_size, polyorder=polyorder)
