mclass-sky
==========
**Multiclass Active Learning Algorithms with Application in Astronomy.**

:Contributors: `Alasdair Tran <http://alasdairtran.com>`_,
               `Cheng Soon Ong <http://www.ong-home.my>`_,
               `Jakub Nabaglo <https://github.com/nbgl>`_,
               `David Wu <https://github.com/davidjwu>`_,
               `Wei Yen Lee <https://weiyen.net>`_
:License: This package is distributed under a a 3-clause ("Simplified" or "New") BSD license.
:Source: `<https://github.com/chengsoonong/mclass-sky>`_
:Doc: `<https://mclearn.readthedocs.io/en/latest/>`_
:Publications: `Combining Active Learning Suggestions <projects/peerjcs16/paper>`_ by Alasdair Tran, Cheng Soon Ong, and Christian Wolf

               `Active Learning with Gaussian Processes <projects/jakub/thesis/nabaglo17photometric-redshift.pdf>`_ by Jakub Nabaglo

               `Photometric Classification with Thompson Sampling <projects/alasdair/thesis/tran15honours-thesis.pdf>`__ by Alasdair Tran

               `Cutting-Plane Methods with Active Learning <projects/david/report/dwu_asc_report_16s2.pdf>`_ by David Wu

.. image:: https://travis-ci.org/chengsoonong/mclass-sky.svg
    :target: https://travis-ci.org/chengsoonong/mclass-sky

.. image:: https://coveralls.io/repos/chengsoonong/mclass-sky/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/chengsoonong/mclass-sky?branch=master

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.58500.svg
   :target: http://dx.doi.org/10.5281/zenodo.58500


Introduction
------------
This repository contains a collection of projects related to active learning
methods with application in astronomy. Click on one of the links below
to go to the directory of a particular project.

1. `Combining Active Learning Suggestions <projects/peerjcs16>`_ by Alasdair Tran, Cheng Soon Ong,
   and Christian Wolf

2. `Active Learning with Gaussian Processes for Photometric Redshift Prediction <projects/jakub>`_

3. `Cutting-plane Methods with Applications in Convex Optimization and Active Learning <projects/david>`_

4. `Photometric Classification with Thompson Sampling <projects/alasdair>`__


mclearn
-------

**mclearn** is a Python package that implement selected multiclass active learning
algorithms, with a focus in astronomical data.

The dependencies are Python 3.4, numpy, pandas, matplotlib, seaborn, ephem, scipy, ipython,
and scikit-learn. It's best to first install the Anaconda distribution for Python 3,
then install **mclearn** using pip::

  pip install mclearn
