import json
import sys

import numpy as np
import sklearn.model_selection

import model

# Import splitter
sys.path.insert(1, '..')
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 50000

MAX_AGP = 5000
STEP = 100


def passive_strategy(agp, train_X):
    assert len(train_X.shape) == 2
    assert train_X.shape[0] >= STEP
    out = np.arange(STEP)
    # print(out)
    return out


def active_strategy(agp, train_X):
    assert len(train_X.shape) == 2
    out = agp.recommend(train_X, STEP)
    # print(out)
    return out


def simulate_active_learning(strategy, repeat_cv,
                             train_X, train_y, test_X, test_y):
    if repeat_cv:
        all_X = train_X[:STEP]
        all_y = train_y[:STEP]
        train_X = train_X[STEP:]
        train_y = train_y[STEP:]

        agp = model.AppxGPModelWithCV()
        agp.fit(all_X, all_y)

        out_scores = [agp.score(test_X, test_y)]
        print('Done', STEP)

        for s in range(2 * STEP, MAX_AGP + 1, STEP):
            recommendations = strategy(agp, train_X)
            all_X = np.append(all_X, train_X[recommendations], 0)
            all_y = np.append(all_y, train_y[recommendations], 0)
            assert all_X.shape[1] == train_X.shape[1]
            assert len(all_y.shape) == 1 and len(all_X.shape) == 2

            train_X = np.delete(train_X, recommendations, 0)
            train_y = np.delete(train_y, recommendations, 0)

            agp.fit(all_X, all_y)

            out_scores.append(agp.score(test_X, test_y))
            print('Done', s)

        return out_scores

    else:
        agp = model.AppxGPModelWithCV()
        agp.fit(train_X[:STEP], train_y[:STEP])
        train_X = train_X[STEP:]
        train_y = train_y[STEP:]

        out_scores = [agp.score(test_X, test_y)]
        print('Done', STEP)

        for s in range(2 * STEP, MAX_AGP + 1, STEP):
            recommendations = strategy(agp, train_X)
            new_train_X = train_X[recommendations]
            new_train_y = train_y[recommendations]
            train_X = np.delete(train_X, recommendations, 0)
            train_y = np.delete(train_y, recommendations, 0)

            agp.add_fit(new_train_X, new_train_y)

            out_scores.append(agp.score(test_X, test_y))
            print('Done', s)

        return out_scores


def get_curve(data, strategy, repeat_cv, label):
    print('Starting ', label)
    X, y = data

    plot_x = list(range(STEP, MAX_AGP+1, STEP))
    plot_ys = []

    kf = sklearn.model_selection.KFold(n_splits=3)
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


SETTINGS = [
        {'strategy': passive_strategy, 'repeat_cv': False, 'label': 'Passive'},# without repeat CV'},
        {'strategy': active_strategy, 'repeat_cv': False, 'label': 'Active'},# without repeat CV'},
        # {'strategy': passive_strategy, 'repeat_cv': True, 'label': 'Passive with repeat CV'},
        # {'strategy': active_strategy, 'repeat_cv': True, 'label': 'Active with repeat CV'}
    ]
def main(path_in, path_out):
    data = splitter.load(path_in)
    # assert data[0].shape[0] > 1000000
    data, _ = splitter.split(data, data[0].shape[0], 0)
    X, _ = data
    X[:,1:] -= X[:,:-1]

    curves = [get_curve(data, **params) for params in SETTINGS]

    with open(path_out, 'w') as f:
        json.dump(curves, f)


if __name__ == '__main__':
    main(*sys.argv[1:3])
