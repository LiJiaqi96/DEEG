import pandas as pd
import numpy as np
from scipy import io

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
