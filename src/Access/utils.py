import pandas as pd
import numpy as np
from scipy import io

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