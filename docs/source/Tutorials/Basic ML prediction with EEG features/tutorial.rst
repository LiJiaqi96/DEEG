Basic ML prediction with EEG features
=====================================

| In this tutorial, we’ll introduce the following characteristics of
  DEEG package:
| 1. Data loading 2. Quality check of dataset 3. Signal banding (delta,
  theta, alpha, beta and gamma frequency bands) 4. Feature extraction 5.
  Prediction with machine learning models built by scikit-learn

**Step 1:** Load deeg and sklearn package, together necessary
dependencies

.. code:: ipython3

    # !pip install deeg
    import deeg
    import sklearn
    import numpy as np

| **Step 2:** Load dataset. Here we used commonly-used DEAP dataset as
  an example
| After loading, the program will outout the shape of data and labels

.. code:: ipython3

    # Specify the directory of your downloaded DEAP dataset below
    data_dir = "/data1/ljq/datasets/EEG/DEAP/"
    deap_dataset = deeg.load_DEAP(data_dir)
    deap_data, deap_label = deap_dataset[0], deap_dataset[1]


.. parsed-literal::

    Shape of data: (1240, 40, 8064)
    Shape of labels: (1240, 4)


| **Step 3:** Check whether the loaded dataset has missing values.
| If missing value (NaN) occurs (“check_result” is not none), you may
  need to check your data.

.. code:: ipython3

    check_result = deeg.process.data_quality_check(deap_data)
    print(check_result=={})


.. parsed-literal::

    *** Data Quality Check ***
    True


| **Step 4:** Filter temporal signal by several bands for further
  feature extraction
| This step will return a dictionary with keys as band name, and values
  as filtered signal

.. code:: ipython3

    deap_data, deap_label = deap_data[:100, ...], deap_label[:100, ...]
    print(deap_data.shape, deap_label.shape)


.. parsed-literal::

    (100, 40, 8064) (100, 4)


.. code:: ipython3

    deap_bands = deeg.process.band(deap_data)

| **Step 5:** Feature extraction
| Extract commonly-used EEG features on all frequency bands

.. code:: ipython3

    deap_features, names = deeg.features.cal_eeg_features(deap_bands["delta"], sf=1000, ws=2000)
    for band in ["theta", "alpha", "beta", "gamma"]:
        temp_features, names = deeg.features.cal_eeg_features(deap_bands[band], sf=1000, ws=2000)
        deap_features = np.concatenate([deap_features, temp_features], axis=-1)
    print(deap_features.shape)


.. parsed-literal::

    (100, 40, 1001) (100, 40, 1)
    (100, 40, 1001) (100, 40, 1)
    (100, 40, 1001) (100, 40, 1)
    (100, 40, 1001) (100, 40, 1)
    (100, 40, 1001) (100, 40, 1)
    (100, 40, 5025)


**Step 6:** Reshape the extracted features and corresponding labels to
(samples, features/labels) format for ML operations

.. code:: ipython3

    deap_features = deap_features.reshape(deap_features.shape[0], -1)
    print(deap_features.shape)


.. parsed-literal::

    [[ 2.24507164e-09  3.11940765e-07  3.03837908e-06 ...  2.65723488e-03
       1.05805435e+01 -7.59288472e+00]
     [ 6.34293586e-07  4.16585084e-06  4.13876008e-05 ...  4.07333451e-03
       2.54002987e+01 -7.78019837e+00]
     [ 2.75466318e-07  1.81367955e-06  1.75321840e-05 ...  3.08742616e-05
       4.47279497e+00 -7.70771464e+00]
     ...
     [ 2.19018476e-06  1.62185397e-05  1.74806438e-04 ... -2.30126623e-03
       3.67409821e+00 -7.64874794e+00]
     [ 2.73488689e-07  1.79152229e-06  1.85823390e-05 ...  8.68680725e-04
       1.21490870e+01 -7.42536039e+00]
     [ 1.36297845e-06  1.02303531e-05  1.11483876e-04 ... -1.92467443e-03
       6.84929673e+00 -7.64153889e+00]]


| **Step 7:** Emotion prediction by ML models built by scikit-learn
| Including train-test split, training and evaluation processes
| For the convenience of illustration, we binarized the 9-level labels
  in to 0(1-4) and 1(5-9)

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    
    deap_label = (deap_label>=5).astype("int")
    X_train, X_test, y_train, y_test = train_test_split(deap_features, deap_label, test_size=0.33, random_state=0)

.. code:: ipython3

    from sklearn.svm import SVC
    
    model = SVC(kernel="rbf")
    model.fit(X_train, y_train[:,0])
    model.score(X_test, y_test[:,0])




.. parsed-literal::

    0.6363636363636364


