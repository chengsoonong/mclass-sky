import json
import sys
import os
import numpy as np
import model

# Import splitter
sys.path.insert(1, '..')
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 50000

MAX_AGP = 5000
START = 500
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

        for s in range(START + STEP, MAX_AGP + 1, STEP):
            recommendations = strategy(agp, train_X)
            all_X = np.append(all_X, train_X[recommendations], 0)
            all_y = np.append(all_y, train_y[recommendations], 0)
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

        for s in range(START + STEP, MAX_AGP + 1, STEP):
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
    (train_X, train_y), (test_X, test_y) = data
    x = list(range(START, MAX_AGP+1, STEP))
    y = simulate_active_learning(strategy, repeat_cv,
                                 train_X[:MAX_AGP], train_y[:MAX_AGP], test_X, test_y)
    print('Finished', label)
    return {'x': x, 'y': y, 'label': label}


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
        {'strategy': passive_strategy, 'repeat_cv': False, 'label': 'Passive without repeat CV'},
        {'strategy': active_strategy, 'repeat_cv': False, 'label': 'Active without repeat CV'},
        {'strategy': passive_strategy, 'repeat_cv': True, 'label': 'Passive with repeat CV'},
        {'strategy': active_strategy, 'repeat_cv': True, 'label': 'Active with repeat CV'}
    ]


def main(path_in, path_out):
    data = splitter.load(path_in)
    data = splitter.split(data, TRAINING_SAMPLES_NUM, TESTING_SAMPLES_NUM)

    for params in SETTINGS:
        curve = get_curve(data, **params)
        json_append(path_out, curve)


if __name__ == '__main__':
    main(*sys.argv[1:3])
