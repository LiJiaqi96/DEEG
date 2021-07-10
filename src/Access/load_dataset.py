"""
Main package for loading existing public datasets:
    1. DEAP
    2. SEED
    3. DREAMER
    4. NeuroMarketing
to numpy array with dimension (p, m, e).
"""
from utils import data_integration
import numpy as np

def load_DEAP(path):
    """
    :param path: path to DEAP dataset
    DEAP dataset contains 32 participant files so the path points to the folder that stores those files.
    :return: numpy array
    """
    if path[-1] != "/":
        path += "/"
    mat_list_1 = [path + 's0' + str(x) for x in range(1, 9)]
    mat_list_2 = [path + 's' + str(x) for x in range(10, 33)]
    data, labels = data_integration(mat_list_1 + mat_list_2)
    return data, labels