API Reference
=================

This contains the API of all the functions of mclearn. The content is the
same as what you would get from the docstrings, namely the
parameters and the return type of each function.
If this is the first time you're using mclearn, you might like to
instead go through the User Guide for an in-depth explanation of the algorithms.


Classifiers
--------------------------------
.. currentmodule:: mclearn.classifier
.. automodule:: mclearn.classifier
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
    predict_unlabelled_objects



Active Learner
--------------------------------
.. currentmodule:: mclearn.active
.. automodule:: mclearn.active
.. autosummary::
    :nosignatures:
    :toctree: generated/

    BaseActive
    ActiveLearner
    ActiveBandit
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
    compute_A
    compute_F
    compute_pool_variance
    pool_variance_h
    compute_pool_entropy
    pool_entropy_h



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
    fetch_sloan_data
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
    balanced_train_test_split
    csv_to_hdf


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

