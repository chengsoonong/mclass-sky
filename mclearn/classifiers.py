""" Some general standard classifier routines for astronomical data. """

import mclearn
import pickle
import gc
import numpy as np
from sklearn import metrics
from pandas import DataFrame, MultiIndex
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def train_classifier(data, feature_names, class_name, train_size, test_size,
    random_state=None, coords=True, recall_maps=True, classifier=None, correct_baseline=None,
    store_covariances=False):
    """ Standard classifier routine.

        Parameters
        ----------
        X_train : array, shape = [n_samples, n_features]
            Feature matrix of the training examples.

        X_test : array, shape = [n_samples, n_features]
            Feature matrix of the test examples.

        y_train : array, shape = [n_samples]
            The array of class labels corresponding to the training examples.

        y_test : array, shape = [n_samples]
            The array of class labels corresponding to the test examples.

        classifier : Classifier
            An initialised scikit-learn Classifier object.

        Returns
        -------
        classifier : Classifier
            Return the trained scikit-learn Classifier object.

    """

    X_train, X_test, y_train, y_test = mclearn.balanced_train_test_split(
        data, feature_names, class_name, train_size, test_size, random_state=random_state)

    if not classifier:
        classifier = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight='auto', random_state=random_state)

    if coords:
        coords_train = X_train[:, 0:2]
        coords_test = X_test[:, 0:2]
        X_train = X_train[:, 2:]
        X_test = X_test[:, 2:]


    correct_boolean, confusion_test = print_classification_result(X_train, X_test, y_train,
        y_test, recall_maps, classifier, correct_baseline, store_covariances)

    return correct_boolean, confusion_test


def print_classification_result(X_train, X_test, y_train, y_test,
    recall_maps=True, classifier=None, correct_baseline=None, store_covariances=False):
    """
    """

    # train and test
    classifier.fit(X_train, y_train, store_covariances=store_covariances)
    y_pred_test = classifier.predict(X_test)
    confusion_test = metrics.confusion_matrix(y_test, y_pred_test)
    balanced_accuracy = mclearn.balanced_accuracy_expected(confusion_test)

    # put confusion matrix in a DataFrame
    classes = ['Galaxy', 'Quasar', 'Star']
    pred_index = MultiIndex.from_tuples(list(zip(['Predicted'] * 3, classes)))
    act_index = MultiIndex.from_tuples(list(zip(['Actual'] * 3, classes)))
    confusion_features_df = DataFrame(confusion_test, columns=pred_index, index=act_index)

    # display results
    class_names = ['Galaxy', 'Star', 'Quasar']
    print('Here\'s the confusion matrix:')
    display(confusion_features_df)
    print('The balanced accuracy rate is {:.2%}.'.format(balanced_accuracy))
    print('Classification report:')
    print(classification_report(y_test, y_pred_test, class_names, digits=4))

    correct_boolean = y_test == y_pred_test

    # plot the recall maps
    if recall_maps:
        if correct_baseline is None:
            print('Recall Maps of Galaxies, Stars, and Quasars, respectively:')
            mclearn.plot_recall_maps(coords_test, y_test, y_pred_test, class_names, output,
                correct_boolean, vmin=0.7, vmax=1, mincnt=None, cmap=plt.cm.YlGn)
        else:
            print('Recall Improvement Maps of Galaxies, Stars, and Quasars, respectively:')
            correct_diff = correct_boolean.astype(int) - correct_baseline.astype(int)
            mclearn.plot_recall_maps(coords_test, y_test, y_pred_test, class_names, output,
                correct_diff, vmin=-0.2, vmax=+0.2, mincnt=20, cmap=plt.cm.RdBu)

    return correct_boolean, confusion_test


def learning_curve(sample_sizes, data, feature_cols, class_col, classifier, random_state=None,
    normalise=True, degree=1, pickle_path='learning_curve.pickle'):
    """
    """

    lc_accuracy_test = []

    for i in sample_sizes:
        gc.collect()
        # split data into test set and training set (balanced classes are not enforced)
        X_train, X_test, y_train, y_test = mclearn.balanced_train_test_split(
            data, feature_cols, class_col, train_size=i, test_size=200000, random_state=random_state)
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state*2)
        X_test, y_test = shuffle(X_test, y_test, random_state=random_state*3)

        if normalise:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if degree > 1:
            poly_features = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
            X_train = poly_features.fit_transform(X_train)
            X_test = poly_features.transform(X_test)

        # train the classifier
        classifier.fit(X_train, y_train)

        # apply classifier on test set
        y_pred_test = classifier.predict(X_test)
        confusion_test = metrics.confusion_matrix(y_test, y_pred_test)
        lc_accuracy_test.append(mclearn.performance.balanced_accuracy_expected(confusion_test))

    # pickle learning curve
    with open(pickle_path, 'wb') as f:
        pickle.dump(lc_accuracy_test, f, pickle.HIGHEST_PROTOCOL) 
    
    return lc_accuracy_test


def compute_all_learning_curves(data, feature_cols, target_col):
    """
    """

    # define the range of the sample sizes
    sample_sizes = np.concatenate((np.arange(100, 1000, 100), np.arange(1000, 10000, 1000),
                                             np.arange(10000, 100001, 10000), [200000, 300000]))
    sample_sizes_per_class = (sample_sizes / 3).astype(int)

    # due to limited memory, we need to stop earlier when using polynomial kernel of degree 3
    sample_sizes_small = np.concatenate((np.arange(100, 1000, 100), np.arange(1000, 10000, 1000),
                                             np.arange(10000, 40000, 10000)))
    sample_sizes_small_per_class = (sample_sizes_small / 3).astype(int)

    # initialise the classifiers
    svm_rbf = SVC(kernel='rbf', gamma=0.01, C=100, cache_size=2000)
    svm_sigmoid = SVC(kernel='sigmoid', gamma=0.001, C=1000, cache_size=2000)
    svm_poly = LinearSVC(C=0.1, loss='squared_hinge', penalty='l1', dual=False, multi_class='ovr',
                         fit_intercept=True, random_state=21)
    logistic = LogisticRegression(penalty='l1', dual=False, C=1, multi_class='ovr', solver='liblinear', random_state=21)
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='auto', random_state=21)

    # train SVM with RBF kernel (this will take a few hours)
    lc_svm_rbf = learning_curve(sample_sizes_per_class, data, feature_cols, target_col, svm_rbf, random_state=2,
        normalise=True, pickle_path='pickle/04_learning_curves/lc_svm_rbf.pickle')

    # train SVM with polynomial kernel of degree 2
    lc_svm_poly_2 = learning_curve(sample_sizes_per_class, data, feature_cols, target_col, svm_poly, degree=2,
        random_state=2, normalise=True, pickle_path='pickle/04_learning_curves/lc_svm_poly_2.pickle')

    # train SVM with polynomial kernel of degree 3
    lc_svm_poly_3 = learning_curve(sample_sizes_small_per_class, data, feature_cols, target_col, svm_poly, degree=3,
        random_state=2, normalise=True, pickle_path='pickle/04_learning_curves/lc_svm_poly_3.pickle')

    # train logistic regression with polynomial kernel of degree 2
    lc_logistic_2 = learning_curve(sample_sizes_per_class, data, feature_cols, target_col, logistic, degree=2,
        random_state=2, normalise=True, pickle_path='pickle/04_learning_curves/lc_logistic_2.pickle')

    # train logistic regression with polynomial kernel of degree 3
    lc_logistic_3 = learning_curve(sample_sizes_small_per_class, data, feature_cols, target_col, logistic, degree=3,
        random_state=2, normalise=True, pickle_path='pickle/04_learning_curves/lc_logistic_3.pickle')

    # train a random forest
    lc_forest = learning_curve(sample_sizes_per_class, data, feature_cols, target_col, forest,
        random_state=2, normalise=True, pickle_path='pickle/04_learning_curves/lc_forest.pickle')



def grid_search(X, y, classifier, param_grid, train_size=300, test_size=300, clf_name=None,
    report=True):
    """
    """

    cv = StratifiedShuffleSplit(y, n_iter=5, train_size=train_size,
        test_size=test_size, random_state=17)
    grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    if not clf_name:
        clf_name = str(classifier.__class__)

    if report:
        print("The best parameters for {} are {} with a score of {:.2%}.".format(
            clf_name, grid.best_params_, grid.best_score_))
    return grid



def grid_search_svm_rbf(X, y, train_size=300, test_size=300, fig_path='heat.pdf'):
    """
    """

    # define search domain
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid_svm = dict(gamma=gamma_range, C=C_range)

    # run grid search
    classifier = SVC(kernel='rbf')
    grid = mclearn.grid_search(X, y, classifier, param_grid_svm, clf_name='SVM RBF')
    scores = mclearn.reshape_grid_socres(grid.grid_scores_, len(C_range), len(gamma_range))

    # plot scores in a heat map
    fig = plt.figure(figsize=(10, 5))
    ax = mclearn.plot_validation_accuracy_heatmap(scores, x_range=gamma_range,
        y_range=C_range, y_label='$C$', x_label='$\gamma$', power10='both')
    fig.savefig(fig_path, bbox_inches='tight')

    # pickle scores
    with open('pickle/04_learning_curves/grid_scores_svm_rbf.pickle', 'wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 



def grid_search_svm_sigmoid(X, y, train_size=300, test_size=300, fig_path='heat.pdf'):
    """
    """

    # define search domain
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid_svm = dict(gamma=gamma_range, C=C_range)

    # run grid search
    classifier = SVC(kernel='sigmoid')
    grid = mclearn.grid_search(X, y, classifier, param_grid_svm, clf_name='SVM Sigmoid')
    scores = mclearn.reshape_grid_socres(grid.grid_scores_, len(C_range), len(gamma_range))

    # plot scores in a heat map
    fig = plt.figure(figsize=(10, 5))
    ax = mclearn.plot_validation_accuracy_heatmap(scores, x_range=gamma_range,
        y_range=C_range, y_label='$C$', x_label='$\gamma$', power10='both')
    fig.savefig(fig_path, bbox_inches='tight')

    # pickle scores
    with open('pickle/04_learning_curves/grid_scores_svm_sigmoid.pickle', 'wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 



def grid_search_svm_poly_degree(X, y, param_grid, degree=2, train_size=300, test_size=300):
    """
    """

    # transform features to polynomial space
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    # run grid search on various combinations
    classifier = LinearSVC(dual=False, fit_intercept=True, multi_class='ovr',
        loss='squared_hinge', penalty='l1', random_state=13)
    grid1 = mclearn.grid_search(X_poly, y, classifier, param_grid,
        train_size=train_size, test_size=test_size, report=False)

    classifier = LinearSVC(dual=False, fit_intercept=True, multi_class='ovr',
        loss='squared_hinge', penalty='l2', random_state=13)
    grid2 = mclearn.grid_search(X_poly, y, classifier, param_grid,
        train_size=train_size, test_size=test_size, report=False)

    classifier = LinearSVC(dual=True, fit_intercept=True, multi_class='ovr',
        loss='hinge', penalty='l2', random_state=13)
    grid3 = mclearn.grid_search(X_poly, y, classifier, param_grid,
        train_size=train_size, test_size=test_size, report=False)

    classifier = LinearSVC(fit_intercept=True, multi_class='crammer_singer',
        random_state=13)
    grid4 = mclearn.grid_search(X_poly, y, classifier, param_grid,
        train_size=train_size, test_size=test_size, report=False)

    # construct the scores
    scores_flat = grid1.grid_scores_ + grid2.grid_scores_ + grid3.grid_scores_ + grid4.grid_scores_

    return scores_flat



def grid_search_svm_poly(X, y, train_size=300, test_size=300, fig_path='heat.pdf'):
    """
    """

    # define search domain
    C_range = np.logspace(-6, 6, 13)
    param_grid = dict(C=C_range)

    scores_1 = mclearn.grid_search_svm_poly_degree(
        X, y, param_grid, degree=1, train_size=train_size, test_size=test_size)
    scores_2 = mclearn.grid_search_svm_poly_degree(
        X, y, param_grid, degree=2, train_size=train_size, test_size=test_size)
    scores_3 = mclearn.grid_search_svm_poly_degree(
        X, y, param_grid, degree=3, train_size=train_size, test_size=test_size)

    scores = scores_1 + scores_2 + scores_3
    scores = mclearn.reshape_grid_socres(scores, 12, len(C_range))

    ylabels = ['Degree 1, OVR, Squared Hinge, L1-norm',
               'Degree 1, OVR, Squared Hinge, L2-norm',
               'Degree 1, OVR, Hinge, L2-norm',
               'Degree 1, Crammer-Singer',
               'Degree 2, OVR, Squared Hinge, L1-norm',
               'Degree 2, OVR, Squared Hinge, L2-norm',
               'Degree 2, OVR, Hinge, L2-norm',
               'Degree 2, Crammer-Singer',
               'Degree 3, OVR, Squared Hinge, L1-norm',
               'Degree 3, OVR, Squared Hinge, L2-norm',
               'Degree 3, OVR, Hinge, L2-norm',
               'Degree 3, Crammer-Singer']

    # plot scores on heat map
    fig = plt.figure(figsize=(10, 5))
    ax = mclearn.plot_validation_accuracy_heatmap(scores, x_range=C_range, x_label='$C$', power10='x')
    plt.yticks(np.arange(0, 12), ylabels)
    fig.savefig(fig_path, bbox_inches='tight')

    # pickle scores
    with open('pickle/04_learning_curves/grid_scores_svm_poly.pickle', 'wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 



def grid_search_logistic_degree(X, y, param_grid, degree=2, train_size=300, test_size=300):
    """
    """

    # transform features to polynomial space
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    # run grid search
    classifier = LogisticRegression(fit_intercept=True, dual=False, solver='liblinear',
        multi_class='ovr', penalty='l1', random_state=51)
    grid1 = mclearn.grid_search(X_poly, y, classifier, param_grid, report=False)

    classifier = LogisticRegression(fit_intercept=True, dual=False, solver='liblinear',
        multi_class='ovr', penalty='l2', random_state=51)
    grid2 = mclearn.grid_search(X_poly, y, classifier, param_grid, report=False)

    classifier = LogisticRegression(fit_intercept=True, dual=False, solver='lbfgs',
        multi_class='multinomial', penalty='l2', random_state=51)
    grid3 = mclearn.grid_search(X_poly, y, classifier, param_grid, report=False)

    # construct the scores
    scores_flat = grid1.grid_scores_ + grid2.grid_scores_ + grid3.grid_scores_

    return scores_flat


def grid_search_logistic(X, y, train_size=300, test_size=300, fig_path='heat.pdf'):
    """
    """

    # define search domain
    C_range = np.logspace(-6, 6, 13)
    param_grid = dict(C=C_range)

    scores_1 = mclearn.grid_search_logistic_degree(
        X, y, param_grid, degree=1, train_size=train_size, test_size=test_size)
    scores_2 = mclearn.grid_search_logistic_degree(
        X, y, param_grid, degree=2, train_size=train_size, test_size=test_size)
    scores_3 = mclearn.grid_search_logistic_degree(
        X, y, param_grid, degree=3, train_size=train_size, test_size=test_size)

    scores = scores_1 + scores_2 + scores_3
    scores = mclearn.reshape_grid_socres(scores, 9, len(C_range))

    ylabels = ['Degree 1, OVR, L1-norm',
               'Degree 1, OVR, L2-norm',
               'Degree 1, Multinomial, L2-norm',
               'Degree 2, OVR, L1-norm',
               'Degree 2, OVR, L2-norm',
               'Degree 2, Multinomial, L2-norm',
               'Degree 3, OVR, L1-norm',
               'Degree 3, OVR, L2-norm',
               'Degree 3, Multinomial, L2-norm']

    # plot scores on heat map
    fig = plt.figure(figsize=(10, 5))
    ax = mclearn.plot_validation_accuracy_heatmap(scores, x_range=C_range, x_label='$C$', power10='x')
    plt.yticks(np.arange(0, 9), ylabels)
    fig.savefig(fig_path, bbox_inches='tight')

    # pickle scores
    with open('pickle/04_learning_curves/grid_scores_logistic.pickle', 'wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 