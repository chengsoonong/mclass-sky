import numpy as np


def recommend_uncertainty(batch=1):
    def recommender(predictor, X, n=1, X_preprocessed=False):
        assert n >= batch > 0
        assert n % batch == 0

        if n == batch:
            std = predictor.predict(X, return_mean=False,
                                       return_std=True,
                                       X_preprocessed=X_preprocessed)
            best = np.argpartition(std, std.shape[0] - n)[-n:]
            return best

        else:
            if not X_preprocessed:
                X = predictor.preprocess_X(X)
            else:
                X = X.copy()

            predictor = predictor.clone(clone_mean=False,
                                        deep_clone_transform=False)
            retval = np.empty(n, dtype=int)
            indices = np.array(range(X.shape[0]))

            first = True
            for i in range(0, n, batch):
                if not first:
                    predictor.fit(xs, fit_mean=False,
                                      fit_variance=True,
                                      reset=False,
                                      X_preprocessed=True)

                    indices[best] = indices[-batch:]
                    X[best] = X[-batch:]

                    indices = indices[:-batch]
                    X = X[:-batch]
                first = False

                std = predictor.predict(X, return_mean=False,
                                           return_std=True,
                                           X_preprocessed=True)

                best = np.argpartition(std, std.shape[0] - batch)[-batch:]
                xs = X[best]
                retval[i:i + batch] = indices[best]

            return retval
    return recommender


def recommend_random(shuffle=True, random_state=None):
    if not shuffle and random_state is not None:
        raise ValueError()

    if shuffle:
        def recommender(predictor, X, n=1, X_preprocessed=False):
            assert 0 < n <= X.shape[0]
            if random_state is not None:
                np.random.seed(random_state)
            return np.random.choice(X.shape[0], n, replace=False)

    else:
        def recommender(predictor, X, n=1, X_preprocessed=False):
            assert 0 < n <= X.shape[0]
            return np.arange(n, dtype=int)

    return recommender


def recommend_prediction_change(mu, sigma,
                                norm=2,
                                normalize=False,
                                num_X_samples=100,
                                num_y_samples=10,
                                random_state=None):
    assert sigma.shape[0] == sigma.shape[1] == mu.shape[0]

    if norm is 'inf':
        def recommender(predictor, X, n=1, X_preprocessed=False):
            raise NotImplementedError()

    elif 0 < norm:
        def recommender(predictor, X, n=1, X_preprocessed=False):
            assert X_preprocessed or sigma.shape[0] == X.shape[1]
            assert n <= X.shape[0]

            if random_state is not None:
                np.random.seed(random_state)

            X_samples = np.random.multivariate_normal(mu, sigma,
                                                      size=num_X_samples,
                                                      check_valid='ignore')
            X_samples = predictor.preprocess_X(X_samples)
            preds = predictor.predict(X_samples, X_preprocessed=True)

            if normalize:
                normalization = 1 / scipy.stats.multivariate_normal(
                    X_samples, mean=mu, cov=sigma)
            else:
                normalization = np.ones(num_X_samples)

            y_samples = np.random.normal(size=num_y_samples)

            if not X_preprocessed:
                X = predictor.preprocess_X(X)

            scores = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                x = X[i:i+1]

                y_mean, y_std = predictor.predict(x,
                                                  return_std=True,
                                                  X_preprocessed=True)
                x_y_samples = y_samples * y_std
                x_y_samples += y_mean

                cumul = 0
                for y in x_y_samples:
                    y = np.array((y,))  # Does this need to be an array?
                    new_predictor = predictor.clone(clone_variance=False,
                                                    deep_clone_transform=False)
                    new_predictor.fit(x, y, X_preprocessed=True, reset=False)
                    new_preds = new_predictor.predict(X_samples,
                                                      X_preprocessed=True)
                    new_preds -= preds
                    new_preds = np.fabs(new_preds, out=new_preds)
                    new_preds = np.power(new_preds, norm, out=new_preds)
                    new_preds *= normalization
                    cumul += new_preds.sum()

                scores[i] = cumul

            best = np.argpartition(scores, scores.shape[0] - n)[-n:]
            return best

    else:
        raise ValueError()

    return recommender
