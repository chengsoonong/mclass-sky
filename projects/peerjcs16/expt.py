import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from time import time
from mclearn.experiment import ActiveExperiment, load_results, save_results
from mclearn.tools import log
warnings.filterwarnings('ignore')  # Ignore annoying numpy warnings


binary_uci_sets = ['ionosphere', 'magic', 'miniboone', 'pima', 'sonar', 'wpbc']
multi_uci_sets = ['iris', 'wine', 'glass', 'pageblocks', 'vehicle']
uci_sets = binary_uci_sets + multi_uci_sets
datasets =  sorted(uci_sets + ['sdss'])

methods_al =  ['baseline', 'margin', 'w-margin', 'confidence',
               'w-confidence', 'entropy', 'w-entropy',
               'qbb-margin', 'qbb-kl']
methods_bandits = ['thompson', 'ocucb', 'klucb', 'exp++']
methods_rank = ['borda', 'geometric', 'schulze']
methods_no_passive = methods_al + methods_bandits + methods_rank
methods = ['passive'] + methods_no_passive

measures = ['f1', 'accuracy', 'mpba']

def run_expt(X, y, dataset, methods, scale=True):
    log(dataset, end='')
    for method in methods:
        log('.', end='')
        expt = ActiveExperiment(X, y, dataset, method, scale, n_splits=10, n_jobs=10)
        expt.run_policies()

    # Only run COMB on binary datasets
    if dataset in binary_uci_sets:
        expt = ActiveExperiment(X, y, dataset, 'comb', scale, n_splits=10, n_jobs=10)
        expt.run_policies()

    expt = ActiveExperiment(X, y, dataset, None, scale, n_splits=10, n_jobs=10)
    expt.run_asymptote()
    log('')

def train():
    for dataset in uci_sets:
        data_path = os.path.join('data', dataset + '.csv')
        data = pd.read_csv(data_path)
        X, y = data.iloc[:, 1:], data['target']
        run_expt(X, y, dataset, methods)

    if 'sdss' in datasets:
        data_path = os.path.join('data', 'sdss.h5')
        data = pd.read_hdf(data_path, 'sdss')
        class_idx = data.columns.get_loc('class')
        X, y = data.iloc[:, (class_idx+1):], data['class']
        run_expt(X, y, 'sdss', methods, False)

    # Calculate the asymptotes
    for (i, dataset) in enumerate(datasets):
        maximum = {}
        for measure in measures:
            asymptote_measure = 'asymptote_' + measure
            max_measure = 'max_' + measure
            results = {}
            for method in methods:
                results[method] = load_results(dataset, method, measure, True)
            results['asymptote'] = load_results(dataset, 'asymptote', asymptote_measure, True)
            maximum[max_measure] = results['asymptote']
            for method in methods:
                maximum[max_measure] = max(maximum[max_measure], max(results[method]))
        save_results(dataset, 'max', maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run active learning experiment.')
    parser.add_argument('--train', action='store_true', help='Run the actual experiments.')
    parser.add_argument('--plot', action='store_true', help='Generate result plots.')

    args = parser.parse_args()
    if args.train:
        train()
    elif args.plot:
        pass
