#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deeg
import os

path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
data, labels = deeg.load_NeuroMarketing(path)
print("Data shape: {}".format(data.shape))


# In[2]:


import deeg
import os

path = os.path.abspath('.') + '/data.csv'
data = deeg.load_data(path)

