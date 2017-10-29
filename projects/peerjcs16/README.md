# Combining Active Learning Suggestions

This directory contains the notebooks that are used to replicate
the experiments in the paper 'Combining Active Learning Suggestions' by
Alasdair Tran, Cheng Soon Ong, and Christian Wolf.

## Installation

We recommend using the Anaconda distribution for Python 3. You also
need to have the `mclearn` installed. To install it from source:

```
git clone https://github.com/chengsoonong/mclass-sky.git
cd mclass-sky; python setup.py develop
```

## Replicating the Experiments

There are two main notebooks used to replicate the experiments.

1. [UCI Datasets](uci_datasets.ipynb):
   Run this notebook first to generate process the UCI data and put them
   in the right format for the experiments. In addition to the UCI data,
   you also need to have the SDSS dataset, which can be manually downloaded
   from [Zendo](http://dx.doi.org/10.5281/zenodo.58500).

2. [Active Learning Suggestions](active_learning_suggestions.ipynb):
   The second notebook contains the code used to replicate the experiments
   and produce all the plots used in the paper. Note that with the larger
   datasets, it can take up to a day to run. For testing purposes, we recommend
   running with the small datasets such as `wine`. If you would like to
   reproduce the plots used in the paper without re-running the experiments,
   set the `RUN_EXPERIMENTS` constant to `False` in the notebook and
   pull the latest result data from the subdirectory `results` (which is
   a git submodule).
