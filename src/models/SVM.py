# coding:UTF-8
'''
使用提取的 de_LDS 特征进行情感分类，分类器使用 SVM，快速验证。
Created by Xiao Guowen.
'''
from utils.tools import build_extracted_features_dataset
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import scipy.io as scio
import os
DE_feature_path = '../data/ExtractedFeatures/'

def get_labels(label_path):
    '''
        得到15个 trials 对应的标签
    :param label_path: 标签文件对应的路径
    :return: list，对应15个 trials 的标签，1 for positive, 0 for neutral, -1 for negative
    '''
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]
def get_frequency_band_idx(frequency_band):
    '''
        获得频带对应的索引，仅对 ExtractedFeatures 目录下的数据有效
    :param frequency_band: 频带名称，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return idx: 频带对应的索引
    '''
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]

def build_extracted_features_dataset(folder_path, feature_name, frequency_band):
    '''
        将 folder_path 文件夹中的 ExtractedFeatures 数据转化为机器学习常用的数据集，区分开不同 trial 的数据
        ToDo: 增加 channel 的选择，而不是使用所有的 channel
    :param folder_path: ExtractedFeatures 文件夹对应的路径
    :param feature_name: 需要使用的特征名，如 'de_LDS'，'asm_LDS' 等，以 de_LDS1 为例，维度为 62 * 235 * 5，235为影片长度235秒，每秒切分为一个样本，62为通道数，5为频带数
    :param frequency_band: 需要选取的频带，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
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
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_features_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                     verify_compressed_data_integrity=False)
                    subject_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for trials in range(1, 16):
                        feature_vector_list = []
                        label_list = []
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T  # 转置后，维度为 N * 62, N 为影片长度
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
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict




def svm(feature_path,feature_type,frequency_band):
    '''
        按照 SEED 数据集原始论文中的 SVM 计算方式测试准确率和方差，每个 experiment 分开计算，取其中 9 个 trial 为训练集，6 个 trial 为测试集
    :param folder_path: ExtractedFeatures 文件夹路径
    :return None:
    '''
    # 样本加载
    de_LDS_feature_dict, de_LDS_label_dict = build_extracted_features_dataset(feature_path ,feature_type,frequency_band)
    accuracy = 0
    for key in de_LDS_feature_dict.keys():
        print('当前处理到 experiment_{}'.format(key))
        cur_feature = de_LDS_feature_dict[key]
        cur_label = de_LDS_label_dict[key]
        train_feature = []
        train_label = []
        test_feature = []
        test_label = []
        for trial in cur_feature.keys():
            if int(trial) < 10:
                print(trial)
                train_feature.extend(cur_feature[trial])
                train_label.extend(cur_label[trial])
                
            else:
                test_feature.extend(cur_feature[trial])
                test_label.extend(cur_label[trial])
        # 定义 svm 分类器
        print(len(train_feature))
        svc_classifier = svm.SVC(C=0.8, kernel='rbf')
        svc_classifier.fit(train_feature, train_label)
        pred_label = svc_classifier.predict(test_feature)
        print(confusion_matrix(test_label, pred_label))
        print(classification_report(test_label, pred_label))
        cur_accuracy = svc_classifier.score(test_feature, test_label)
        accuracy += cur_accuracy
        print('当前 experiment 的 accuracy 为：{}'.format(cur_accuracy))

    print('所有 experiment 上的平均 accuracy 为：{}'.format(accuracy / len(de_LDS_feature_dict.keys())))



#An example to use SVM
"""
svm(feature_path,feature_type,frequency_band)
para：
feature_path,包括不同feature种类
feature_type,特征种类，例如‘de_LDS’
frequency_band,特征频带，例如‘gamma’

"""