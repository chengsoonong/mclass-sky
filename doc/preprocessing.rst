Data Preprocessing
==================



Normalisation
-------------
If a dataset contains features with vastly different scales,
it is advisible to normalise the features first. There are a few option:

* Normalise the features to zero mean and unit variance.
* Normalise the features to unit variance.
* Normalise the features to unit interval.


Balanced Train-Test Split
-------------------------
Often, the class distribution in a dataset is not balanced.
For example, in the SDSS dataset, we have three times as many
galaxies as quasars. To correct for this bias, we might want
to select a balanced training and test set. This is achieved
by :py:func:`~mclearn.preprocessing.balanced_train_test_split`.
