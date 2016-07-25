""" Active learning policies. """

from abc import ABC, abstractmethod
from .schulze import SchulzeAggregator

class Policy(ABC):
    """ Abstract base class for a policy.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.
    """

    def __init__(self, pool, labels, classifier, n_candidates, random_state=None):
        self.pool = pool
        self.labels = labels
        self.classifier = classifier
        self.n_candidates
        self.pool_size = len(self.pool)

        if type(random_state) is RandomState:
            self.seed = random_state
        else:
            self.seed = RandomState(random_state)

    @abstractmethod
    def select(self):
        pass

    def add(self, index, label):
        """ Add a newly obtained label to the labelled pool. """
        self.labels[index] = label

    def _sample(self):
        """ Sample from the unlabelled pool. """

        candidate_mask = self.labels.mask

        if 0 < self.n_candidates < np.sum(candidate_mask):
            unlabelled_index = np.where(candidate_mask)[0]
            candidate_index = self.seed.choice(unlabelled_index, self.sample_size, replace=False)
            candidate_mask = np.zeros(self.pool_size, dtype=bool)
            candidate_mask[candidate_index] = True

        return candidate_mask


class SingleSuggestion(Policy):
    def __init__(self, pool, labels, arm):
        super().__init__(pool, labels)
        self.arm = arm

    def select(self):
        candidate_mask = self._sample()
        predictions = self.classifier.predict(self.pool[candidate_mask])
        best_candidates = self.arm.select(candidate_mask, predictions)
        return best_candidates


class MultipleSuggestions(Policy):
    def __init__(self, pool, labels, arms):
        super().__init__(pool, labels)
        self.arms = arms
        self.n_arms = len(arms)


class ActiveBandit(MultipleSuggestions):
    def __init__(self, pool, labels, arms):
        super().__init__(pool, labels, arms)
        self.time_step = 0
        self.T = np.zeros(self.n_arms)
        self.reward_history = []
        self.T_history = [self.T.copy()]

    def _select_from_arm(self):
        candidate_mask = self._sample()
        predictions = self.classifier.predict(self.pool[candidate_mask])
        best_candidates = self.arms[self.selected_arm].select(candidate_mask, predictions)
        return best_candidates

    @abstractmethod
    def receive_reward(self, reward):
        pass

    @abstractmethod
    def history(self):
        pass


class ThompsonSampling(ActiveBandit):
    """ Thompson Sampling. """

    def __init__(self, pool, labels, arms, mu=0, sigma=0.02, likelihood_sigma=0.02):
        super().__init__(pool, labels, arms)
        self.mu = np.full(self.n_arms, mu, dtype=np.float64)
        self.sigma = np.full(self.n_arms, sigma, dtype=np.float64)
        self.likelihood_sigma = np.full(self.n_arms, likelihood_sigma, dtype=np.float64)

        self.mu_history = [self.mu.copy()]
        self.sigma_history = [self.sigma.copy()]


    def select(self):
        """ Select the next candidate. """
        # take a sample of rewards from the current prior of heuristics
        sample_rewards = self.seed.normal(self.mu, self.sigma)
        self.selected_arm = np.argmax(sample_rewards)
        return self._select_from_arm(self)


    def receive_reward(self, reward):
        mu_0 = self.mu[self.selected_arm]
        sigma_0 = self.sigma[self.selected_arm]
        sigma = self.likelihood_sigma[self.selected_arm]

        self.mu[self.selected_arm] = (mu_0 * sigma + reward * sigma_0) / (sigma + sigma_0)
        self.sigma[self.selected_arm] = (sigma_0 * sigma) / (sigma + sigma_0)
        self.T[self.selected_arm] += 1

        # store results in history
        self.mu_history.append(self.mu.copy())
        self.sigma_history.append(self.sigma.copy())
        self.T_history.append(self.T.copy())
        self.reward_history.append(reward)


    def history(self):
        history = {}
        history['mu'] = np.asarray(mu_history)
        history['sigma'] = np.asarray(sigma_history)
        history['T'] = np.asarray(T_history)
        history['reward'] = np.asarray(reward_history)
        return history


class UCUCB(ActiveBandit):
    def __init__(self, pool, labels, arms, alpha=3, psi=2):
        super().__init__(pool, labels, arms)
        self.alpha = 3
        self.psi = 2
        self.mu = np.zeros(self.n_arms)
        self.sum_mu = np.zeros(self.n_arms)
        self.mu_history = [self.mu.copy()]

    def select(self):
        self.time_step += 1

        if self.time_step <= self.n_arms:
            self.selected_arm = self.time_step - 1
            return self._select_from_arm()

        else:
            ucb = self.mu + np.sqrt((self.alpha / self.T) *
                                     np.log(self.psi * self.horizon / self.time_step))
            self.selected_arm = np.argmax(ucb)
            return self._select_from_arm()


    def receive_reward(self, reward):
        # update empirical estimate of the reward and the times an arm is selected
        self.sum_mu[self.selected_arm] += reward
        self.T[self.selected_arm] += 1
        self.mu[self.selected_arm] = self.sum_mu[self.selected_arm] / self.T[self.selected_arm]

        # store results in history
        self.mu_history.append(self.mu.copy())
        self.T_history.append(self.T.copy())
        self.reward_history.append(reward)

    def history(self):
        history = {}
        history['mu'] = np.asarray(mu_history)
        history['T'] = np.asarray(T_history)
        history['reward'] = np.asarray(reward_history)
        return history




class ActiveAggregator(MultipleSuggestions):
    def __init__(self, pool, labels, arms, n_best_candidates):
        super().__init__(pool, labels, arms)
        self.n_best_candidates = n_best_candidates

    def select(self):
        candidate_mask = self._sample()
        predictions = self.classifier.predict(self.pool[candidate_mask])

        voters = [arm.select(candidate_mask, predictions, self.n_candidates) for arm in arms]
        voters = np.array(voters)
        best_candidates = self._aggregate_votes(voters)
        return best_candidates

    @abstractmethod
    def _aggregate_votes(self, voters):
        pass


class BordaAggregator(ActiveAggregator):
    def _aggregate_votes(self, voters):
        max_score = len(voters[0])
        points = {}

        # accumulate the points for each candidate
        for voter in voters:
            for idx, candidate in enumerate(voter):
                points[candidate] = points.get(candidate, 0) + max_score - idx

        # sort the candidates and return the most popular one(s)
        rank = sorted(points, key=points.__getitem__, reverse=True)
        return rank[:self.n_best_candidates]


class GeometricAggregator(ActiveAggregator):
    def _aggregate_votes(self, voters):
        max_score = len(voters[0])
        points = {}

        # accumulate the points for each candidate
        for voter in voters:
            for idx, candidate in enumerate(voter):
                points[candidate] = points.get(candidate, 1) * (max_score - idx)

        # sort the candidates and return the most popular one(s)
        rank = sorted(points, key=points.__getitem__, reverse=True)
        return rank[:n_best_candidates]
