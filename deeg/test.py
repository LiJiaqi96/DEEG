import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# from process import denoise
# from deeg.check import check_nan
# from deeg.band import band
# from deeg.sampling import sampling
# from deeg.segment import segment
from deeg.features import cal_eeg_features

data_dir = "D:/EEG/data_preprocessed_matlab/"

s01 = sio.loadmat(data_dir+"s01.mat")
s01_data, s01_label = s01["data"], s01["labels"]

single_s01_data = s01_data[0,0,:]
print(s01_data.shape, single_s01_data.shape)
print(len(s01_data.shape), len(single_s01_data.shape))

# ----- Extract all features -----
eeg_features, feature_list = cal_eeg_features(s01_data, 1000, 2000)
print(eeg_features.shape)
# from deeg.features import power_spectral_density
# a = power_spectral_density(single_s01_data, 1000, 2000)
# print(a)


# ----- Process: segment -----
# segmented_array = segment(single_s01_data, length=10, overlap=3)
#
# print(segmented_array[0].shape, segmented_array[1].shape)
# print(single_s01_data[:20])
# print(segmented_array.shape)




# ----- Process: sampling -----
# sampled_array = sampling(s01_data, 10, 1000)
# print(sampled_array.shape)

# ----- Process: band -----
# ts = band(single_s01_data)
# print(ts)
# print(ts["alpha"].shape)

# ----- Process: check -----
# print(check_nan(s01_data))

# ----- Features: DE -----
# def cal_de(s):
#     return (1 / 2) * np.log(2 * np.pi * np.e * np.std(s) ** 2)
#
# print(cal_de(single_s01_data))
#
#
# b = np.ones((2,3,2))
# c = np.apply_along_axis(cal_de, axis=-1, arr=s01_data)
# print(c.shape)
# print(c)
# print(cal_de(s01_data[39,39]))
# -----

