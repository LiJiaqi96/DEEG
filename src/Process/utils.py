import os

import numpy as np
import scipy.io as sio
from scipy import interpolate
from keras.preprocessing.sequence import pad_sequences
### 论文中下采样至200Hz
def downSample(data, originLen, targetLen):
    '''
    input: Signal ([nbSamples, nbChannels, length])
    originLen: Original Length
    targetLen: Target Length
    '''

    gap = originLen // targetLen

    idx = np.arange(0, originLen, gap)
    downSampled = data[:, :, idx]

    return downSampled


def create_dir(dataset_name, data_type):
    '''
    生成数据相应文件夹
    input:
    : dataset_name: dataset name
    : data_type: 
    '''
    if not os.path.exists(f"./{dataset_name}"):
        os.mkdir(f"./{dataset_name}")
    if not os.path.exists(f"./{dataset_name}/{data_type}/"):
        os.mkdir(f"./{dataset_name}/{data_type}/")
    for i in range(15):
        if not os.path.exists(f"./{dataset_name}/{data_type}/subject_{i}"):
            os.mkdir(f"./{dataset_name}/{data_type}/subject_{i}")
        for j in range(3):
            if not os.path.exists(f"./{dataset_name}/{data_type}/subject_{i}/section_{j}"):
                os.mkdir(
                    f"./{dataset_name}/{data_type}/subject_{i}/section_{j}")


def seqToMap(seq):
    '''
    将电信号序列转换成3D map格式
    '''
    nSample, _, depth = seq.shape
    map3d = np.zeros([nSample, 9, 9, depth])
    for sampleIdx in range(nSample):  # samples
        # Row 0-1
        map3d[sampleIdx][0][3] = seq[sampleIdx][0]
        map3d[sampleIdx][0][4] = seq[sampleIdx][1]
        map3d[sampleIdx][0][5] = seq[sampleIdx][2]
        map3d[sampleIdx][1][3] = seq[sampleIdx][3]
        map3d[sampleIdx][1][5] = seq[sampleIdx][4]

        # Row 2-5
        for m in range(2, 7):
            for n in range(9):
                map3d[sampleIdx][m][n] = seq[sampleIdx][(m - 2) * 9 + n + 5]

        # Row 7
        for m in range(1, 8):
            map3d[sampleIdx][7][m] = seq[sampleIdx][m + 49]

        # Row 8
        for m in range(2, 7):
            map3d[sampleIdx][8][m] = seq[sampleIdx][m + 55]
    return map3d


def interpolation(map3d, tgt_sideLen):
    '''
    对3D map进行插值操作
    input
    : map3d: 3D map input
    : tgt_sideLen: target sideLen
    '''
    nSample, raw_sideLen, raw_sideLen, depth = map3d.shape
    map3d = map3d.transpose(0, 3, 1, 2)

    interpolatedMap = np.zeros([nSample, depth, tgt_sideLen, tgt_sideLen])
    for sampleIdx in range(nSample):
        for d in range(depth):
            x = np.arange(raw_sideLen)
            y = np.arange(raw_sideLen)
            z = map3d[sampleIdx][d]

            f = interpolate.interp2d(x, y, z, kind='cubic')
            xnew = np.linspace(0, raw_sideLen - 1, tgt_sideLen)
            ynew = np.linspace(0, raw_sideLen - 1, tgt_sideLen)
            interpolatedMap[sampleIdx][d] = f(xnew, ynew)
    return interpolatedMap


def sectionMapConstruction(dataset_name, data_type, subject_id, section_id, section_path):
    '''
    
    '''
    nb_trials = 15

    section_feature = sio.loadmat(section_path)
    if data_type == 'feature':
        # from de_LDS1 to de_LDS15
        keys = [f'de_LDS{i+1}' for i in range(15)]
    elif data_type == 'temporal':
        # from eeg_1 to eeg_15
        keys = list(section_feature.keys())[3:]
    trial_output = []
    for trial_id in range(nb_trials):
        # get trial name
        trial_feature = section_feature[keys[trial_id]]
        if data_type == 'feature':
            trial_feature = trial_feature.swapaxes(0, 1)
        elif data_type == 'temporal':
            length = 200
            trial_feature = trial_feature[:, :-
                                          1].reshape([trial_feature.shape[0], -1, length])
            trial_feature = trial_feature.transpose(1, 0, 2)
            trial_feature = downSample(trial_feature, length, 25)

        create_dir(dataset_name, data_type)
        map3d = seqToMap(trial_feature)
        interpolatedMap = interpolation(map3d, tgt_sideLen=32)
        outputMap = np.array(
            interpolatedMap, dtype=np.float32).transpose(0, 2, 3, 1)
        outputMap = np.expand_dims(outputMap, axis=-1)
        outputMap = np.mean(outputMap, axis=0)
        #print('output Map shape: ', outputMap.shape)
        trial_output.append(outputMap)
        np.save(
            f'./{dataset_name}/{data_type}/subject_{subject_id}/section_{section_id}/trial_{trial_id}.npy', outputMap)
        # print('Successfully saved',f'./{dataset_name}/{data_type}/subject_{subject_id}/section_{section_id}/trial_{trial_id}.npy')

    np.save(f'./{dataset_name}/{data_type}/subject_{subject_id}/section_{section_id}_data.npy',trial_output)
    output = np.array(trial_output,dtype=object)
    # output = pad_sequences(outputMap, padding='post')
    return output




def create_train_test_dir(dataset_name, main_folder, data_type):
    '''
    生成数据相应文件夹
    input:
    : dataset_name: dataset name
    : data_type: 
    '''
    # create main folder
    if not os.path.exists(f"./{main_folder}"):
        os.mkdir(f"./{main_folder}")

    # create train and test subfolder
    if not os.path.exists(f"./{dataset_name}/{main_folder}/train/"):
        os.mkdir(f"./{dataset_name}/{main_folder}/train/")
    if not os.path.exists(f"./{dataset_name}/{main_folder}/test/"):
        os.mkdir(f"./{dataset_name}/{main_folder}/test/")
    if not os.path.exists(f"./{dataset_name}/{main_folder}/train/{data_type}/"):
        os.mkdir(f"./{dataset_name}/{main_folder}/train/{data_type}/")
    if not os.path.exists(f"./{dataset_name}/{main_folder}/test/{data_type}/"):
        os.mkdir(f"./{dataset_name}/{main_folder}/test/{data_type}/")
    
    for i in range(15):
        if not os.path.exists(f"./{dataset_name}/{main_folder}/train/{data_type}/subject_{i}"):
            os.mkdir(f"./{dataset_name}/{main_folder}/train/{data_type}/subject_{i}")
        if not os.path.exists(f"./{dataset_name}/{main_folder}/test/{data_type}/subject_{i}"):
            os.mkdir(f"./{dataset_name}/{main_folder}/test/{data_type}/subject_{i}")
        # for j in range(3):
        #     if not os.path.exists(f"./{dataset_name}/{main_folder}/train/{data_type}/subject_{i}/section_{j}"):
        #         os.mkdir(f"./{dataset_name}/{main_folder}/train/{data_type}/subject_{i}/section_{j}")
        #     if not os.path.exists(f"./{dataset_name}/{main_folder}/test/{data_type}/subject_{i}/section_{j}"):
        #         os.mkdir(f"./{dataset_name}/{main_folder}/test/{data_type}/subject_{i}/section_{j}")



if __name__=='__main__':
    from natsort import natsorted
    import re
    dataset_name = 'SEED'
    # Spectral Feature Preprocess
    parent_path = os.path.abspath(os.path.join(os.getcwd(), ""))
    temporal_root_path = os.path.join(parent_path, f'{dataset_name}/Preprocessed_EEG')
    feature_root_path = os.path.join(parent_path, f'{dataset_name}/ExtractedFeatures')

    feature_files_list = os.listdir(feature_root_path)
    feature_files_path_list = [os.path.join(
        feature_root_path, name) for name in feature_files_list]
    feature_files_path_list = natsorted(feature_files_path_list)

    # regular expression to search .mat file
    skip_list = []
    for i in feature_files_path_list:
        if not re.match(r'.+/[0-9]+_[0-9]+.mat', i):
            print('skipped: ', i)
            skip_list.append(i)

    for i in skip_list:
        feature_files_path_list.remove(i)

    feature_sub_file_list = []
    for i in range(0, len(feature_files_path_list), 3):
        feature_sub_file_list.append(feature_files_path_list[i: i+3])
    subject_id = 1
    section_id = 1
    map_i_feature = sectionMapConstruction(dataset_name, 'feature', subject_id,
                               section_id, feature_sub_file_list[subject_id][section_id])
    print(map_i_feature.shape)
