mclearn
=======
**Multiclass Active Learning Algorithms with Application in Astronomy.**

:Author: `Alasdair Tran <http://alasdairtran.com>`_
:License: This package is distributed under a a 3-clause ("Simplified" or "New") BSD license.
:Source: `<https://github.com/alasdairtran/mclearn>`_
:Docs: `<http://pythonhosted.org/mclearn/>`_

.. image:: https://travis-ci.org/chengsoonong/mclass-sky.svg?branch=master
    :target: https://travis-ci.org/chengsoonong/mclass-sky
       
Introduction
------------------------------
**mclearn** is a Python package that implement selected multiclass active learning
algorithms, with a special focus in astronomical data.



Installation
------------------------------
The presequisites are Python 3.4, numpy, pandas, matplotlib, and scikit-learn.
To install using pip::

   pip install mclearn



Example Notebooks
------------------------------
As a quick start, have a look at the following Jupyter notebooks and see what mclearn
can do:

* `About the SDSS Dataset`_
* `Dimensionality Reduction in SDSS`_
* `Feature Selection of Photometric Measurements`_
* `Implementing Standard Classifiers`_
* `Active Learning with Contextual Bandits`_
* `Active Learning with Logistic Regression`_
* `Making Photometric Prediction`_



.. _About the SDSS Dataset:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/about_sdss.ipynb
.. _Dimensionality Reduction in SDSS:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/dimensionality_reduction.ipynb
.. _Feature Selection of Photometric Measurements:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/feature_selection.ipynb
.. _Implementing Standard Classifiers:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/standard classifiers.ipynb
.. _Active Learning with Contextual Bandits:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/contextual_bandits.ipynb
.. _Active Learning with Logistic Regression:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/logistic_active_learning.ipynb
.. _Making Photometric Prediction:
   http://nbviewer.ipython.org/github/alasdairtran/mclearn/blob/master/examples/predicting_unknowns.ipynb



Datasets
--------

The datasets are too big for GitHub. They can be downloaded from the
`NICTA
filestore <http://filestore.nicta.com.au/mlrg-data/astro/sdss_dr7_photometry.csv.gz>`__.
