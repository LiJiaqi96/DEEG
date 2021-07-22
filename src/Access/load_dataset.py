"""
Main package for loading existing public datasets:
    1. DEAP
    2. SEED
    3. DREAMER
    4. NeuroMarketing
to numpy array with dimension (p, m, e).
"""
from scipy.io import loadmat

from utils import data_integration, sort_NeuroMarketing, get_labels, get_frequency_band_idx
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

def load_SEED(folder_path, feature_name, frequency_band):
    '''
    :param folder_path: directory of ExtractedFeatures
    :param feature_name: feature name, for example 'de_LDS', 'asm_LDS' etc. Take de_LDS1 as an example: the demension is (62, 235, 5), 62 for 62 channels, 235 for 235 seconds and 5 for 5 different frequency bands.
    :param frequency_band: the input band name: 'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return numpy array of data and labels
    '''
    frequency_idx = get_frequency_band_idx(frequency_band)
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    feature_vector_dict = {}
    label_dict = {}
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('Currently process: {}, total progress: {}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_features_dict = loadmat(os.path.join(folder_path, file_name),
                                                     verify_compressed_data_integrity=False)
                    subject_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for trials in range(1, 16):
                        feature_vector_list = []
                        label_list = []
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T  # dimensions: N * 62, N is the length of video
                        feature_vector_list.extend(_ for _ in cur_feature)
                        for _ in range(len(cur_feature)):
                            label_list.append(labels[trials - 1])
                        feature_vector_trial_dict[str(trials)] = feature_vector_list
                        label_trial_dict[str(trials)] = label_list
                    feature_vector_dict[subject_name] = feature_vector_trial_dict
                    label_dict[subject_name] = label_trial_dict
                else:
                    continue
    except FileNotFoundError as e:
        print('Loading Error: {}'.format(e))

    data = []
    labels = []
    for experiment in feature_vector_dict.keys():
        for trial in feature_vector_dict[experiment].keys():
            data.extend(feature_vector_dict[experiment][trial])
            labels.extend(label_dict[experiment][trial])
    data = np.array(data)
    labels = np.array(labels).T
    return data, labels
