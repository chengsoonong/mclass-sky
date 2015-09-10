mclearn
=======
**Multiclass Active Learning Algorithms with Application in Astronomy.**

:Contributors: `Alasdair Tran <http://alasdairtran.com>`_
:License: This package is distributed under a a 3-clause ("Simplified" or "New") BSD license.
:Source: `<https://github.com/chengsoonong/mclass-sky>`_
:Doc: `<https://mclearn.readthedocs.org/en/latest/>`_

.. image:: https://travis-ci.org/alasdairtran/mclearn.svg
    :target: https://travis-ci.org/alasdairtran/mclearn

.. image:: https://coveralls.io/repos/alasdairtran/mclearn/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/alasdairtran/mclearn?branch=master


       
Introduction
------------
**mclearn** is a Python package that implement selected multiclass active learning
algorithms, with a focus in astronomical data. For a quick overview of how
**mclearn** works, have a look at the `Getting Started`_ notebook.


Installation
------------
The dependencies are Python 3.4, numpy, pandas, matplotlib, seaborn, ephem, scipy, ipython,
and scikit-learn. It's best to first install the Anaconda distribution for Python 3,
then install **mclearn** using pip::

  pip install mclearn



Experiments
-----------
Datasets
~~~~~~~~
Throughout the experiments, we will be using the dataset from the Sloan Digital Sky Survey.
Due to their size, the following datasets are not included in this repo: ::

  projects/alasdair/data/
  │   sdss.h5
  │   sdss_dr7_photometry_source.csv.gz
  │   sdss_full.h5  
  │   sdss_subclass.h5

The above datasets (except for `sdss_full.h5`)
can be downloaded from the `NICTA filestore <http://filestore.nicta.com.au/mlrg-data/astro/>`__.

Notebooks
~~~~~~~~~

1. `Getting the SDSS Dataset`_
    We provide instruction on how to obtain the SDSS dataset from the Sloan SkySever.
    We then clean up the data and convert the `csv` files to HDF5 for quicker reading.

2. `Exploratory Data Analysis`_
    To get a feel for the data, we plot the distributions of the classes (Galaxy, Quasar, Star).
    We will see that the data is quite unbalanced, with three times as many galaxies as quasars.
    A distinction is made between photometry and spectroscopy. We also use PCA to reduce the
    data down to two dimensions.

3. `Colours and Dust Extinction`_
    To see which combinations of magnitudes and colours should be used, we use randomised 
    logistic regression to assign a score to each feature. We then select the 17 best features, 
    which included the original magnitudes plus a few colours. Dust extinction is a potential
    problem in photometry, so we compare three sets of reddening corrections (SFD98, SF11, and
    W14) to see which set is best at removing the bias. The SF11 set performs the best, however
    the difference is very small.

4. `Learning Curves`_
    To see how random sampling performs, we construct learning curves for SVMs, Logistic
    Regression, and Random Forest. A grid search with a 5-fold cross validation
    is performed to choose the best hyperparameters for the SVM and Logistic Regression.
    We also do a polynomial transformation of degree 2 and 3 on the features.

5. `Logistic Active Learning`_
    Let's see if we can be smarter at choosing our training set. Various active learning
    heuristics are looked at, including uncertainty sampling, query by bagging, and
    minimising the entropy and the variance of the example pool.

6. `Predicting Unlabelled Objects`_
    We predict the classes of the 800,000 million unlabelled objects using a random
    forest.

6. `VST ATLAS Dataset`_
    We run active learning on the VST ATLAS dataset.

7. `Bandit Active Learning`_
    We examine active learning in a multi-arm bandit setting.

.. _Getting Started:
   projects/alasdair/notebooks/getting_started.ipynb
.. _Getting the SDSS Dataset:
   projects/alasdair/notebooks/01_getting_sdss_dataset.ipynb
.. _Exploratory Data Analysis:
   projects/alasdair/notebooks/02_exploratory_analysis.ipynb
.. _Colours and Dust Extinction:
   projects/alasdair/notebooks/03_colours_and_dust_extinction.ipynb
.. _Learning Curves:
   projects/alasdair/notebooks/04_learning_curves.ipynb
.. _Logistic Active Learning:
   projects/alasdair/notebooks/05_logistic_active_learning.ipynb
.. _Predicting Unlabelled Objects:
   projects/alasdair/notebooks/06_predicting_unlabelled_objects.ipynb
.. _VST ATLAS Dataset:
   projects/alasdair/notebooks/07_vstatlas.ipynb
.. _Bandit Active Learning:
   projects/alasdair/notebooks/08_bandits.ipynb
