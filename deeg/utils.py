import numpy as np
import re
from scipy import io
from scipy.io import loadmat
from numpy import genfromtxt


# ======Utils for dataset======
"""
Utils for loading DEAP Dataset
"""
def data_partition(data):
    X = data['data']
    y = data['labels']
    return X,y

def data_integration(data_list):
    data, labels = data_partition(io.loadmat(data_list[0]))
    for i in range(1, len(data_list)):
        X,y = data_partition(io.loadmat(data_list[i]))
        data = np.concatenate((data, X), axis = 0)
        labels = np.concatenate((labels, y), axis = 0)
    print("Shape of data: {}".format(data.shape))
    print("Shape of labels: {}".format(labels.shape))
    return data, labels

"""
Utils for loading NeuroMarketing Dataset
"""
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_NeuroMarketing(v_list):
    return sorted(v_list, key=str2int)

"""
Utils for loading SEED Dataset
"""
def get_frequency_band_idx(frequency_band):
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]

def get_labels(label_path):
    return loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]

"""
Utils for generic loading
"""
def read_csv(fname):
    return genfromtxt(fname, delimiter=",")

