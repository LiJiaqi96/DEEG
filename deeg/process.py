import numpy as np
from scipy import signal


def data_quality_check(array):
    """
    Check whether the input signal contains NaN, print relevant information
    input:
        array: numpy array. The temporal signal with 1d or with multiple dimensions
    return:
        None or dict. Dict stores indicator of Nan/Abnormal values.
            If there is no NaN in signal, return None, else return the index where NaN occurs
    """
    print("*** Data Quality Check ***")

    def check_nan(array):
        return np.isnan(array).any()

    def check_abnormal(array):
        """
        Todo: Define what kinds of EEG data are abnormal in some kinds?
        """
        return None

    if not check_nan(array) and check_abnormal(array):
        print("*** Data Quality Check Passed ***")
        return None
    else:
        # return format: tuple(array([a,b,c,...]), array([j,k,l,...]), ...)
        # Each element in tuple indicates the index list
        dict = {}
        if check_nan(array):
            print("Nan Values appear in data")
            print("The percentage of Nan Values is {val}%").format(val=sum(np.isnan(array)) / array.shape[0])
            print("The location is at {tuple}").format(tuple=np.where(np.isnan(array)))
            dict["nan"] = np.where(np.isnan(array))
        if check_abnormal(array):
            #Todo: add abnormal values' locations to dict.
            pass
        return dict


def savgol(array, window_size=5, polyorder=2):
    """
    Apply the Savitzky-Golay fiklter to smooth signals.
    We support both 1d array input and DEEG standard input (p,m,e).
    For the later format, we calculate DE on dim "e" and return a (p,m) numpy array.

    Input
    -------
    array: numpy array or list
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
        # creat weighting coefficients
        x = []
        for i in range(2 * size + 1):
            m = i - size
            row = [m ** j for j in range(rank)]
            x.append(row)
        x = np.mat(x)
        return x

    def SG(array, window_size, polyorder):
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        if polyorder >= window_size:
            raise ValueError("polyorder must be less than window_lengthd")

        m = int((window_size - 1) / 2)
        odata = np.array(array).tolist()
        for i in range(m):
            odata.insert(0, odata[0])
            odata.insert(len(odata), odata[len(odata) - 1])

        # creat X matrix
        x = create_x(m, polyorder)
        # calculate the weighting coefficients
        b = (x * (x.T * x).I) * x.T
        a0 = b[m]
        a0 = a0.T
        # calculate the filtered signal
        filtered_signal = []
        for i in range(len(array)):
            y = [odata[i + j] for j in range(window_size)]
            y1 = np.mat(y) * a0
            y1 = float(y1)
            filtered_signal.append(y1)
        return np.array(filtered_signal)
    if len(np.array(array).shape) == 1:
        return SG(array, window_size, polyorder)
    else:
        return np.apply_along_axis(SG, axis=-1, arr=array, window_size=window_size, polyorder=polyorder)


def filter(array, name="delta", order=5):
    """
    A band-pass filter to suppress signal out of frequency band
    input:
        array: numpy array. The input temporal signal with 1d
        name: str. Specify the filter type commonly used in EEG analysis. (delta, theta, alpha, beta, gamma)
        order: int. The order of Butterworth filter. Default is 5
    return:
        ts: numpy array. Filtered signal in temporal domain
    """
    if name == "delta":
        band = [1,3]
    elif name == "theta":
        band = [4,7]
    elif name == "alpha":
        band = [8,13]
    elif name == "beta":
        band = [14,30]
    elif name == "gamma":
        band = [31,50]
    else:
        raise(Exception("Invalid filter name!"))
    sos = signal.butter(order, band, 'bp', fs=1000, output='sos')
    ts = signal.sosfilt(sos, array)
    return ts


def band(array):
    """
    Filter temporal signal by all the 5 filters commonly used in EEG analysis
    input:
        signal: numpy array. The input temporal signal. 1d or with multiple dimensions
    return
        ts_dict: dictionary. Keys are filter name and values are filtered signal in temporal domain.
    """
    ts_dict = {}
    if len(array.shape) < 2:
        for name in ["delta","theta","alpha","beta","gamma"]:
            ts_dict[name] = filter(array)
    else:
        for name in ["delta","theta","alpha","beta","gamma"]:
            ts_dict[name] = np.apply_along_axis(filter, -1, array, name)
    return ts_dict


def segment(array, length=10, overlap=1, label=None):
    """
    Generate the signal segments to augment sample size.
    input:
        array: numpy array. The input temporal signal. 1d or with multiple dimensions
        length: int. The length of segments. Default is 10, which means each signal segment contains 10 data points.
        overlap: int. The overlap length between neighboring segments. Default is 1.
        label: numpy array. If label is not None, duplicate labels according to the data
    return:
        segmented_array: numpy array with shape (p,m,s,e) or (s,e), depending on the input. s is the number of segments.
        duplicated_labels: numpy array with shape (p,m,s,e) or (s,e). If input label is None do not return.
    """
    if len(array.shape) < 2:
        segmented_array = [array[i:i+length] for i in range(0, len(array), length-overlap)]
        if len(segmented_array[-1]) != length:
            segmented_array = segmented_array[:-1]
        if label is None:
            return np.array(segmented_array)
        else:
            return np.array(segmented_array), np.expand_dims(label, 0).repeat(len(segmented_array), axis=0)
    else:
        segmented_array = [array[:,:,i:i+length] for i in range(0, array.shape[-1], length-overlap)]
        if segmented_array[-1].shape[-1] != length:
            segmented_array = segmented_array[:-1]
        if label is None:
            return np.array(segmented_array).transpose(1,2,0,3)   # transpose numpy array to make (p,m,s,e) format
        else:
            return np.array(segmented_array).transpose(1,2,0,3), np.expand_dims(label, -2).repeat(segmented_array.shape[-2], axis=-2)


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
        return array[:, :, offset::interval]