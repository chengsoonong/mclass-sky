API Reference
=================

This contains the API of all the functions of mclearn. The content is the
same as what you would get from the docstrings, namely the
parameters and the return type of each function.
If this is the first time you're using mclearn, you might like to
instead go through the User Guide for an in-depth explanation of the algorithms.

As a convenience, all functions listed below can be accessed directly from the
top-level module.


Active Learner
--------------------------------
.. currentmodule:: mclearn.active_learner
.. automodule:: mclearn.active_learner
.. autosummary::
    :nosignatures:
    :toctree: generated/

    active_learn
    compute_accuracy


Active Learning Heuristics
--------------------------
.. currentmodule:: mclearn.heuristics
.. automodule:: mclearn.heuristics
.. autosummary::
    :nosignatures:
    :toctree: generated/

    random_h
    entropy_h
    margin_h
    qbb_margin_h
    qbb_kl_h


Performance Measures
-------------------------------------
.. currentmodule:: mclearn.performance
.. automodule:: mclearn.performance
.. autosummary::
    :nosignatures:
    :toctree: generated/

    naive_accuracy
    get_beta_parameters
    convolve_betas
    balanced_accuracy_expected
    beta_sum_pdf
    beta_avg_pdf
    beta_sum_cdf
    beta_avg_pdf
    beta_avg_inv_cdf
    recall
    precision


Photometric Data
---------------------------------
.. currentmodule:: mclearn.photometry
.. automodule:: mclearn.photometry
.. autosummary::
    :nosignatures:
    :toctree: generated/

    dust_extinction_w14



Data Preprocessing
------------------------------
.. currentmodule:: mclearn.preprocessing
.. automodule:: mclearn.preprocessing
.. autosummary::
    :nosignatures:
    :toctree: generated/

    normalise_z
    normalise_unit_var
    normalise_01
    draw_random_sample
    balanced_train_test_split


Visualisations
-------------------------------
.. currentmodule:: mclearn.viz
.. automodule:: mclearn.viz
.. autosummary::
    :nosignatures:
    :toctree: generated/

    plot_class_distribution
    plot_scores
    plot_balanced_accuracy_violin
    plot_learning_curve
    plot_hex_map

