import numpy as np

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
