"""
Main package for loading existing public datasets:
    1. DEAP
    2. SEED
    3. DREAMER
    4. NeuroMarketing
to numpy array with dimension (p, m, e).
"""
from utils import data_integration, sort_NeuroMarketing
import numpy as np
import os
import re

def load_DEAP(path):
    """
    :param path: path to DEAP dataset
    DEAP dataset contains 32 participant files so the path points to the folder that stores those files.
    :return: numpy array of data and labels
    """
    if path[-1] != "/":
        path += "/"
    mat_list_1 = [path + 's0' + str(x) for x in range(1, 9)]
    mat_list_2 = [path + 's' + str(x) for x in range(10, 33)]
    data, labels = data_integration(mat_list_1 + mat_list_2)
    return data, labels

def load_NeuroMarketing(path):
    """
    :param path: path to NeuroMarketing dataset
    :return: numpy array of data and labels
    """
    original_name = []
    for file in os.listdir(path + '25-users/'):
        if re.search('\.txt$', file):
            original_name.append(file[:-4])
    name = sort_NeuroMarketing(original_name)
    data = []
    label = []
    for i in name:
        dataFile = open(path + '25-users/' + i + '.txt')
        data.append(np.array([eval(i) for i in dataFile.read().split()]).reshape(512, 14).T)
        labelFile = open(path + 'labels/' + i + '.lab')
        if labelFile.read() == 'Like':
            label.append(1)
        else:
            label.append(0)
    data = np.array(data)
    labels = np.array(label).T
    return data, labels