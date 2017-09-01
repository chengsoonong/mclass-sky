import json
import sys
import os

import numpy as np
import sklearn.model_selection

import model
import recommenders

# Import splitter
sys.path.insert(1, '..')
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 50000

MAX_AGP = 100000
START = 1000
STEP = 100

COMPONENTS = 1000

def simulate_active_learning(strategy, repeat_cv,
                             train_X, train_y, test_X, test_y):
    if repeat_cv:
        # TODO: make this more efficient

        all_X = train_X[:START]
        all_y = train_y[:START]
        train_X = train_X[START:]
        train_y = train_y[START:]

        agp = model.RBFAppxGPWithCVRegressor(n_components=COMPONENTS, random_state=23)
        agp.fit(all_X, all_y, fit_variance=True)

        out_scores = [agp.score(test_X, test_y)]
        print('Done', START)

        for s in range(START + STEP, MAX_AGP + 1, STEP):
            recommendations = strategy(agp, train_X, n=STEP)
            all_X = np.append(all_X, train_X[recommendations], 0)
            all_y = np.append(all_y, train_y[recommendations], 0)
            assert all_X.shape[1] == train_X.shape[1]
            assert len(all_y.shape) == 1 and len(all_X.shape) == 2

            train_X = np.delete(train_X, recommendations, 0)
            train_y = np.delete(train_y, recommendations, 0)

            agp.fit(all_X, all_y, fit_variance=True)

            out_scores.append(agp.score(test_X, test_y))
            print('Done', s)

        return out_scores

    else:
        agp = model.RBFAppxGPWithCVRegressor(n_components=COMPONENTS, random_state=23)
        agp.fit(train_X[:START], train_y[:START], fit_variance=True)
        train_X = train_X[START:]
        train_y = train_y[START:].copy()

        train_Phi = agp.preprocess_X(train_X)
        test_Phi = agp.preprocess_X(test_X)

        out_scores = [agp.score(test_Phi, test_y, X_preprocessed=True)]
        print('Done', START)

        for s in range(START + STEP, MAX_AGP + 1, STEP):
            recommendations = strategy(agp, train_Phi,
                                       n=STEP, X_preprocessed=True)
            new_train_Phi = train_Phi[recommendations]
            new_train_y = train_y[recommendations]

            agp.fit(new_train_Phi, new_train_y, X_preprocessed=True,
                                                reset=False,
                                                fit_variance=True)

            train_Phi[recommendations] = train_Phi[-STEP:]
            train_Phi = train_Phi[:-STEP]

            train_y[recommendations] = train_y[-STEP:]
            train_y = train_y[:-STEP]

            out_scores.append(agp.score(test_Phi, test_y, X_preprocessed=True))
            print('Done', s)

        return out_scores


def get_curve(data, strategy, repeat_cv, label):
    print('Starting ', label)
    X, y = data

    plot_x = list(range(START, MAX_AGP+1, STEP))
    plot_ys = []

    kf = sklearn.model_selection.KFold(n_splits=3, random_state=1234)
    for train_index, test_index in kf.split(X):
        train_X, test_X = X[train_index], X[test_index]

        train_y, test_y = y[train_index], y[test_index]

        plot_y = simulate_active_learning(
            strategy, repeat_cv,
            train_X[:MAX_AGP], train_y[:MAX_AGP], test_X, test_y)
        plot_ys.append(plot_y)
        break

    # TODO: Is this necessary or can json.dump handle np.ndarray?
    avg_plot_y = list(np.mean(plot_ys, axis=0))
    print('Finished', label)
    return {'x': plot_x, 'y': avg_plot_y, 'label': label}


def json_append(fname, entry):
    """If output file does not exist, create it.
    Otherwise append to it (in a list)"""
    if not os.path.isfile(fname):
        a = []
        a.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(fname) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))


SETTINGS = [
        # {'strategy': recommenders.recommend_random(random_state=1), 'repeat_cv': False, 'label': 'Passive'},# without repeat CV'},
        dict(strategy=recommenders.recommend_uncertainty(batch=100), repeat_cv=False, label='Batch 100'),# without repeat CV'},
        # {'strategy': recommenders.recommend_uncertainty(batch=10), 'repeat_cv': False, 'label': 'Batch 10'},# without repeat CV'},
        # {'strategy': recommenders.recommend_uncertainty(batch=1), 'repeat_cv': False, 'label': 'Batch 1'},# without repeat CV'},
        # {'strategy': recommenders.recommend_random(random_state=1), 'repeat_cv': True, 'label': 'CV'},
        # {'strategy': recommenders.recommend_uncertainty(), 'repeat_cv': True, 'label': 'Active with repeat CV'}
    ]


def main(path_in, path_out):
    data = splitter.load(path_in)
    # assert data[0].shape[0] > 1000000
    data, _ = splitter.split(data, data[0].shape[0], 0)
    X, y = data
    # X : u, g, r, i, z
    u, g, r, i, z = X.T
    u_g = u - g
    u_r = u - r
    r_i = r - i
    i_z = i - z

    X[:,0] = r
    X[:,1] = u_g
    X[:,2] = u_r
    X[:,3] = r_i
    X[:,4] = i_z
    data = X, y

    for params in SETTINGS:
        curve = get_curve(data, **params)
        json_append(path_out, curve)


if __name__ == '__main__':
    main(*sys.argv[1:3])
