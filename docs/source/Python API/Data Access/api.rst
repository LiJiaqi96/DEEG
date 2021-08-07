Data Access
===============

------------------

.. py:function:: deeg.load_data(fname, *, preload=False, verbose=None, **kwargs)

   load common data file types (csv, edf, bdf, gdf, vhdr, fif, fif.gz, set, cnt, mff, nxe, hdr, mat, bin, data, sqd, con, ds, txt).

   :param fname: files you are gonna read
   :type fname: str
   :param mne: indicate if the file is mne-supported
   :type mne: str
   :return: numpy array of EEG data

------------------

.. py:function:: deeg.load_DEAP(path)

   Load DEAP dataset and convert it to numpy array with dimension (p, m, e). DEAP dataset contains 32 participant files and the path points to the folder that stores those files.
    
   :param path: path to DEAP dataset
   :type path: str
   :return: numpy array of data and labels

------------------

.. py:function:: deeg.load_NeuroMarketing(path)

   Load NeuroMarketing dataset and convert it to numpy array with dimension (p, m, e). NeuroMarketing dataset contains 25 participant files and the path points to the folder that stores those files.

   :param path: path to NeuroMarketing dataset
   :type path: str
   :return: numpy array of data and labels

------------------

.. py:function:: deeg.load_SEED(folder_path, feature_name, frequency_band)

   Load SEED dataset and convert it to numpy array with dimension (p, m, e). 

   :param folder_path: directory of ExtractedFeatures
   :type folder_path: str
   :param feature_name: feature name, for example 'de_LDS', 'asm_LDS' etc. Take de_LDS1 as an example: the demension is (62, 235, 5), 62 for 62 channels, 235 for 235 seconds and 5 for 5 different frequency bands.
   :type feature_name: str
   :param frequency_band: the input band name: 'delta', 'theta', 'alpha', 'beta', 'gamma'
   :type frequency_band: str
   :return numpy array of data and labels

