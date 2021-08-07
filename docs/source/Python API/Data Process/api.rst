Data Process
===============

Data process is essential in EEG signal analysis. First, original EEG signal may contain missing values and noise, which may disturb or hidden the real information. Also, a lot of research works have shown that the spectrum properties of EEG on certain frequency bands are very important to uncover useful information which cannot be directly quantified by original temporal signal. In addition, the number of samples are relatively small for EEG signal, which may not fully release the power of machine learning. Data augmentation through segmentation and sampling are two major ways to increase sample size in EEG analysis.

So we designed the data process module to provide solutions to the requirements mentioned above. The data process module contains 5 functions:

- process.check(:func:`deeg.process.check`)
- process.smooth(:func:`deeg.process.smooth`)
- process.band(:func:`deeg.process.band`)
- process.segment(:func:`deeg.process.segment`)
- process.sampling(:func:`pdeeg.rocess.sampling`)

------------------

.. py:function:: deeg.process.check(array)

   Check whether the input signal contains NaN, print relevant information

   :param array: the temporal signal with 1d or with multiple dimensions
   :type array: numpy array
   :return: None or dict. Dict stores indicator of Nan/Abnormal values. If there is no NaN in signal, return None, else return the index where NaN occurs.


------------------

.. py:function:: deeg.process.smooth(array, window_size=5, polyorder=2)

   Apply the Savitzky-Golay fiklter to smooth signals. We support both 1d array input and DEEG standard input (p,m,e). For the later format, we calculate DE on "e" dimension and return a (p,m) numpy array.

   :param array: the temporal signal with 1d or with multiple dimensions
   :type array: numpy array
   :param window_size: the length of the filter window. window_size must be a positive odd integer. default is 5.
   :type window_size: int.  window_size must be a positive odd integer. default is 5. 
   :param polyorder: the order of the polynomial used to fit the samples. polyorder must be less than window_length. default is 2.
   :type polyorder: int 
   :return: ndarray of the filtered data, same shape as array.

------------------

.. py:function:: deeg.process.band(array)

   Filter temporal signal by all the 5 filters commonly used in EEG analysis: delta, theta, alpha, beta, gamma.

   :param array: the temporal signal with 1d or with multiple dimensions
   :type array: numpy array
   :return: dictionary. Keys are filter name and values are filtered signal in temporal domain.


------------------

.. py:function:: deeg.process.segment(array, length, overlap=1)

   Generate the signal segments to augment sample size.

   :param array: the temporal signal with 1d or with multiple dimensions
   :type array: numpy array
   :param length: the length of segments. Default is 10, which means each signal segment contains 10 data points.
   :type length: int
   :param overlap: the overlap length between neighboring segments. Default is 1.
   :type overlap: int
   :return: numpy array with shape (p,m,s,e) or (s,e), depending on the input. s is the number of segments.

------------------

.. py:function:: deeg.process.sampling(array, interval=1, offset)

   Down-sample the input signal with certain interval.

   :param array: the temporal signal with 1d or with multiple dimensions
   :type array: numpy array
   :param interval: the interval to sample EEG signal. Default is 1, which means NO down-sampling is applied
   :type interval: int
   :param offset: sampling starts from "offset-th" data point
   :type offset: int
   :return: numpy array of down-sampled signal