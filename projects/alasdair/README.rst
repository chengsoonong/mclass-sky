Photometric Classification with Thompson Sampling
=======
**ANU Honours Thesis 2015**

Introduction
------------
This directory contains code and thesis for `Alasdair Tran's 2015 honours thesis`_
on photometric classification with Thompson sampling.

Datasets
--------
Throughout the experiments, we will be using the dataset from the Sloan Digital Sky Survey.
Due to their size, the following datasets are not included in this repo: ::

  projects/alasdair/data/
  │   sdss.h5
  │   sdss_dr7_photometry_source.csv.gz
  │   sdss_full.h5
  │   sdss_subclass.h5

The main dataset :code:`sdss.h5` is available on `Zenodo <http://dx.doi.org/10.5281/zenodo.58500>`_:


Notebooks
---------

For a quick overview of how
**mclearn** works, have a look at the `Getting Started`_ notebook.

The following nine notebooks accompany Alasdair's thesis on
`Photometric Classification with Thompson Sampling`_.

1. `Dataset Preparation`_
    We provide instruction on how to obtain the SDSS dataset from the Sloan SkySever.
    We then clean up the data and convert the `csv` files to HDF5 for quicker reading.
    We also do some cleaning up of the raw data from the VST ATLAS survey.

2. `Exploratory Data Analysis`_
    To get a feel for the data, we plot the distributions of the classes (Galaxy, Quasar, Star).
    We will see that the data is quite unbalanced, with three times as many galaxies as quasars.
    A distinction is made between photometry and spectroscopy. We also use PCA to reduce the
    data down to two dimensions.

3. `Dust Extinction`_
    TDust extinction is a potential
    problem in photometry, so we compare three sets of reddening corrections (SFD98, SF11, and
    W14) to see which set is best at removing the bias. It turns out that there are no
    significant differences between the three extinction vectors.

4. `Learning Curves`_
    To see how random sampling performs, we construct learning curves for SVMs, Logistic
    Regression, and Random Forest. A grid search with a 5-fold cross validation
    is performed to choose the best hyperparameters for the SVM and Logistic Regression.
    We also do a polynomial transformation of degree 2 and 3 on the features.

5. `Class Proportion Estimation`_
    We predict the classes of the 800,000 million unlabelled SDSS objects using a random
    forest.

6. `Active Learning with SDSS`_
    We look at six active learning heuristics and see how well they perform in the
    SDSS dataset.

7. `Active Learning with VST ATLAS`_
    We look at six active learning heuristics and see how well they perform in the
    VST ATLAS dataset.

8. `Thompson Sampling with SDSS`_
    We know examine the six active learning heuristics under the multi-arm bandit
    setting with Thompson sampling and using the SDSS dataset.

9. `Thompson Sampling with VST ATLAS`_
    We repeat the same Thompson sampling experiment with the VST ATLAS dataset


.. _Alasdair Tran's 2015 honours thesis:
   https://alasdairtran.github.io/mclearn/tran15honours-thesis.pdf
.. _Photometric Classification with Thompson Sampling:
   https://alasdairtran.github.io/mclearn/tran15honours-thesis.pdf
.. _Getting Started:
   notebooks/getting_started.ipynb
.. _Dataset Preparation:
   notebooks/01_dataset_prepration.ipynb
.. _Exploratory Data Analysis:
   notebooks/02_exploratory_analysis.ipynb
.. _Dust Extinction:
   notebooks/03_dust_extinction.ipynb
.. _Learning Curves:
   notebooks/04_learning_curves.ipynb
.. _Class Proportion Estimation:
   notebooks/05_class_proportion_estimation.ipynb
.. _Active Learning with SDSS:
   notebooks/06_active_learning_sdss.ipynb
.. _Active Learning with VST ATLAS:
   notebooks/07_active_learning_vstatlas.ipynb
.. _Thompson Sampling with SDSS:
   notebooks/08_thompson_sampling_sdss.ipynb
.. _Thompson Sampling with VST ATLAS:
   notebooks/09_thompson_sampling_vstatlas.ipynb
