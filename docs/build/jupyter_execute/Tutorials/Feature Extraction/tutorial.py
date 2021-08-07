#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deeg
import os

path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
data, labels = deeg.load_NeuroMarketing(path)
data_Kurt = deeg.features.kurtosis(data)
data_Skew = deeg.features.skewness(data)
data_MAD = deeg.features.mean_absolute_deviation(data)
print("Original data shape: {}".format(data.shape))
print("Data shape after kurtosis feature extraction: {}".format(data_Kurt.shape))
print("Data shape after skewness feature extraction: {}".format(data_Skew.shape))
print("Data shape after mean absolute deviation feature extraction: {}".format(data_MAD.shape))


# In[2]:


import deeg
import os

path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
data, labels = deeg.load_NeuroMarketing(path)
data_DE = deeg.features.differential_entropy(data)
print("Original data shape: {}".format(data.shape))
print("Data shape after differential entropy feature extraction: {}".format(data_DE.shape))


# In[3]:


import deeg
import os

path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
data, labels = deeg.load_NeuroMarketing(path)
data_band = deeg.process.band(data)
data_PSD = dict()
for band in data_band:
    data_PSD[band] = deeg.features.power_spectral_density(data_band[band], 128, 256)
    print("Original data shape of {} band: {}".format(band, data_band[band].shape))
    print("Data shape of {} band after feature extraction: {}".format(band, data_PSD[band].shape))


# In[4]:


import deeg
import os

path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
data, labels = deeg.load_NeuroMarketing(path)
channel_name = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
data_DASM = deeg.features.differential_asymmetry(data, channel_name)
print("Original data shape: {}".format(data.shape))
print("Data shape after feature extraction: {}".format(data_DASM.shape))

