.. Documnet of DEEG documentation master file, created by
   sphinx-quickstart on Thu Aug  5 12:41:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature Extraction
============================================



Temporal Features
-----------------------

-----------------------

.. py:function:: deeg.features.skewness(array)

   Calculate the feature "skewness" of the EEG signal. We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate SKEW on dim "e" and return a (p,m) numpy array.

   :param array: the temporal signal for feature extraction. 1d or with multiple dimensions
   :type array: numpy array
   :return: int or numpy array, depends on the input shape

------------------

.. py:function:: deeg.features.kurtosis(array)

   Calculate the feature "kurtosis" of the EEG signal. We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate KURT on dim "e" and return a (p,m) numpy array.

   :param array: the temporal signal for feature extraction. 1d or with multiple dimensions
   :type array: numpy array
   :return: int or numpy array, depends on the input shape

------------------

.. py:function:: deeg.features.mean_absolute_deviation(array)

   Calculate the mean_absolute_deviation on EEG data

   :param array: the temporal signal for feature extraction. 1d or with multiple dimensions
   :type array: numpy array
   :return: int or numpy array, depends on the input shape

------------------

Frequency Features
-----------------------

-----------------------

.. py:function:: deeg.features.differential_entropy(array)

   Calculate the feature Differential Entropy with hypothesis that processed signal is normal-distributed. We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate DE on dim "e" and return a (p,m) numpy array.

   :param array: the temporal signal for feature extraction. 1d or with multiple dimensions
   :type array: numpy array    
   :return: int or numpy array, depends on the input shape

------------------

.. py:function:: deeg.features.power_spectral_density(array, signal_frequency, window_size)

   Calculate the power spectral density on EEG signal. We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate PSD on dim "e" and return a (p,m) numpy array.

   :param array: the temporal signal for feature extraction. 1d or with multiple dimensions
   :type array: numpy array
   :param signal_frequency: int
   :type signal_frequency: original frequency of the signal
   :param window_size: size of the window
   :type window_size: int
   :return: band tuple or numpy array of power, depends on the input shape

------------------

Mutichannel Features
-----------------------

-----------------------

.. py:function:: deeg.features.rational_asymmetry(array, channel_name)

   Calculate the rational asymmetry, of EEG signal. 

   :param array: the temporal signal(>=2d) for feature extraction.
   :type array: numpy array       
   :param channel_name: the names of all the channels of the EEG signal
   :type channel_name: list or 1d array
   :return: numpy array.

-----------------------

.. py:function:: deeg.features.differential_asymmetry(array, channel_name)
    
   Calculate the differential asymmetry of EEG signal.

   :param array: the temporal signal (>=2d) for feature extraction
   :type array: numpy array 
   :param channel_name: the names of all the channels of the EEG signal
   :type channel_name: list or 1d array
   :return: numpy array

-----------------------

.. py:function:: deeg.features.differential_caudality(array, channel_name)
    
   Calculate the differential caudality of EEG signal.

   :param array: the temporal signal (>=2d) for feature extraction
   :type array: numpy array 
   :param channel_name: the names of all the channels of the EEG signal
   :type channel_name: list or 1d array
   :return: numpy array



