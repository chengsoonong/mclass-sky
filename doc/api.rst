API Reference
=================

This contains the API of all the functions of mclearn. The content is the
same as what you would get from the docstrings, namely the
parameters and the return type of each function.
If this is the first time you're using mclearn, you might like to
instead go through the User Guide for an in-depth explanation of the algorithms.

As a convenience, all functions listed below can be accessed directly from the
top-level module.


Classifiers
--------------------------------
.. currentmodule:: mclearn.classifiers
.. automodule:: mclearn.classifiers
.. autosummary::
    :nosignatures:
    :toctree: generated/

    train_classifier
    print_classification_result
    learning_curve
    compute_all_learning_curves
    grid_search
    grid_search_svm_rbf
    grid_search_svm_sigmoid
    grid_search_svm_poly_degree
    grid_search_svm_poly
    grid_search_logistic_degree
    grid_search_logistic



Active Learner
--------------------------------
.. currentmodule:: mclearn.active_learner
.. automodule:: mclearn.active_learner
.. autosummary::
    :nosignatures:
    :toctree: generated/

    active_learn
    run_active_learning_with_heuristic
    active_learning_experiment


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
    compute_balanced_accuracy


Photometric Data
---------------------------------
.. currentmodule:: mclearn.photometry
.. automodule:: mclearn.photometry
.. autosummary::
    :nosignatures:
    :toctree: generated/

    reddening_correction_sfd98
    reddening_correction_sf11
    reddening_correction_w14
    correct_magnitudes
    compute_colours
    fetch_filter
    fetch_spectrum
    clean_up_subclasses



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
    plot_average_learning_curve
    plot_hex_map
    plot_recall_maps
    plot_filters_and_spectrum
    plot_scatter_with_classes
    reshape_grid_socres
    plot_validation_accuracy_heatmap

