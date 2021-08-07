.. Documnet of DEEG documentation master file, created by
   sphinx-quickstart on Thu Aug  5 12:41:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Data Access
============================================

This tutorial covers the basic EEG methods for data loading. DEEG data structures are based around the numpy array with (p, m, e) structure, where 'p' represents the trail number，'m' represents the channel, and 'e' represents the EEG signal. We also provide reader functions for a wide variety of other data formats. DEEG also has APIs to several publicly available datasets, which DEEG can manage for you.

* Available data formats
csv, edf, bdf, gdf, vhdr, fif, fif.gz, set, cnt, mff, nxe, hdr, mat, bin, data, sqd, con, ds, txt


* Available datasets

==============  =========  =================================================================
Dataset         format     link
==============  =========  =================================================================
DEEP            .dat/.mat  http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
DREAMER         .mat       https://zenodo.org/record/546113#.YQuuJi18hsN
SEED            .mat       http://bcmi.sjtu.edu.cn/home/seed/
NeuroMarketing  .txt       https://drive.google.com/file/d/0B2T1rQUvyyWcSGVVaHZBZzRtTms/view
==============  =========  =================================================================


We’ll start this tutorial by loading the NeuroMarketing dataset, which contains EEG data of 25 subjects when watching 42 photos of different products, along with the photos of 42 products. The :func:`deeg.load_NeuroMarketing` function will automatically load NeuroMarketing dataset and convert the data to our standard (p, m, e) structure.

.. jupyter-execute::
   
   import deeg 
   import os

   path = os.path.abspath('.') + '/Data-EEG-25-users-Neuromarketing/'
   data, labels = deeg.load_NeuroMarketing(path)
   print("Data shape: {}".format(data.shape))

To load DEEP, DREAMER, and SEED datasets, you need to first submit an application for the datasets. Once you get permission , you can download the datasets and load the datasets by :func:`deeg.load_DEEP`, :func:`deeg.load_DREAMER`, :func:`deeg.load_SEED` respectively. These functions are similar as :func:`deeg.load_dataset`, which we show above.

DEEG also supports users to load other data using :func:`deeg.load_data` function. Here is an example of loading a csv data file.

.. jupyter-execute::
   
   import deeg 
   import os

   path = os.path.abspath('.') + '/data.csv'
   data = deeg.load_data(path)
   


