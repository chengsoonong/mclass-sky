""" Policies to select unlabelled candidates among many active learning arms.

    All policies on selecting the next unlabelled instance in the pool for labelling.

    Module structure:
    - Policy
        - SingleSuggestion
        - MultipleSuggestions
            - ActiveBandit
                - ThompsonSampling
                - OCUCB
                - KLUCB
                - EXP3PP
            - ActiveAggregator
"""

# Author: Alasdair Tran
# License: BSD 3 clause

import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState

from mclearn.schulze import _aggregate_votes_schulze

__all__ = ['SingleSuggestion',
           'ThompsonSampling',
           'OCUCB',
           'KLUCB',
           'EXP3PP',
           'ActiveAggregator',
           'COMB']


class Policy(ABC):
    """ Abstract base class for a policy.

        This class cannot be used directly but instead serves as the base class for
        all policies.  Each policy needs to implement the `select` method, which
        return the indices of the pool that we should query next for the label.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, random_state=None,
                 n_candidates=None, n_best_candidates=1):
        self.pool = pool
        self.labels = labels
        self.classifier = classifier
        self.n_candidates = n_candidates if n_candidates is not None else len(self.pool)
        self.n_best_candidates = n_best_candidates
        self.pool_size = len(self.pool)

        if type(random_state) is RandomState:
            self.seed = random_state
        else:
            self.seed = RandomState(random_state)

    @abstractmethod
    def select(self):
        """ Needs to return an array of indices of objects from the pool. """
        pass

    def add(self, index, label):
        """ Add a newly obtained label to the labelled pool and retrain.

            Parameters
            ----------
            index : int or array of ints
                The index or indices of the object(s) that have just been labelled.

            label : object or array of objects
                The label(s) obtained from the oracle.
        """
        self.labels[index] = label
        train_idx = ~self.labels.mask
        self.classifier.fit(self.pool[train_idx], self.labels[train_idx])

    def receive_reward(self, reward):
        """ Receive a reward from the environment and update the policy's parameters. """
        pass

    def history(self):
        """ Return a dictionary containing the history of the policy. """
        return {}

    def _sample(self):
        """ Take a random sample of candidates from the unlabelled pool. """
        candidate_mask = self.labels.mask
        if 0 < self.n_candidates < np.sum(candidate_mask):
            unlabelled_index = np.where(candidate_mask)[0]
            candidate_index = self.seed.choice(unlabelled_index, self.n_candidates, replace=False)
            candidate_mask = np.zeros(self.pool_size, dtype=bool)
            candidate_mask[candidate_index] = True
        return candidate_mask


class SingleSuggestion(Policy):
    """ Select candidates from the pool according to one particular active learning rule.

        This class is a wrapper for a particular active learning rule.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arm : Arm object
            A particular active learning rule. The arm needs to implement the ``select``
            method that returns an array of indices of objects from the pool for
            labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, arm, random_state=None,
                 n_candidates=None, n_best_candidates=1):
        super().__init__(pool, labels, classifier, random_state,
                         n_candidates, n_best_candidates)
        self.arm = arm

    def select(self):
        """ Use the initialised arm to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        candidate_mask = self._sample()
        predictions = self.classifier.predict_proba(self.pool[candidate_mask])
        best_candidates = self.arm.select(candidate_mask, predictions, self.n_best_candidates)
        return best_candidates


class MultipleSuggestions(Policy):
    """ Abstract base class for a policy that takes multiple active learning rules.

        This class cannot be used directly but instead serves as the base class for
        all policies that takes multiple active learning rules.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1):
        super().__init__(pool, labels, classifier, random_state,
                         n_candidates, n_best_candidates)
        self.arms = arms
        self.n_arms = len(arms)


class ActiveBandit(MultipleSuggestions):
    """ Abstract base class for a bandit policy that takes multiple active learning rules.

        This class cannot be used directly but instead serves as the base class for
        all bandit policies that take as input multiple active learning rules.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.time_step = 0
        self.T = np.zeros(self.n_arms)
        self.mu = np.zeros(self.n_arms)
        self.sum_mu = np.zeros(self.n_arms)
        self.reward_history = []
        self.T_history = [self.T.copy()]
        self.mu_history = [self.mu.copy()]

    def _select_from_arm(self):
        """ Use a particular arm to select candidates from the pool. """
        candidate_mask = self._sample()
        predictions = self.classifier.predict_proba(self.pool[candidate_mask])
        best_candidates = self.arms[self.selected_arm].select(
            candidate_mask, predictions, self.n_best_candidates)
        return best_candidates

    def receive_reward(self, reward):
        """ Receive a reward from the environment and updates the policy's prior beliefs.

            Parameters
            ----------
            reward : float
                The reward from the environment should be a good proxy for the decrease
                in the generalisation error of the classifier.
        """
        # update empirical estimate of the reward and the times an arm is selected
        self.sum_mu[self.selected_arm] += reward
        self.T[self.selected_arm] += 1
        self.mu[self.selected_arm] = self.sum_mu[self.selected_arm] / self.T[self.selected_arm]

        # store results in history
        self.mu_history.append(self.mu.copy())
        self.T_history.append(self.T.copy())
        self.reward_history.append(reward)

    def history(self):
        """ Return a dictionary containing the history of the policy.

            Returns
            -------
            history : dict
                The dictionary contains the following keys: mu, T, and reward.
                The corresponding value of each key is an array containing the state
                in each time step.
        """
        history = {}
        history['mu'] = np.array(self.mu_history)
        history['T'] = np.array(self.T_history)
        history['reward'] = np.array(self.reward_history)
        return history


class BaselineCombiner(ActiveBandit):
    """ Baseline bandit combiner.

        At each step, we randomly select an arm, ignoring the rewards.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def select(self):
        """ Select an arm at random.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        self.selected_arm = self.seed.choice(self.n_arms)
        return self._select_from_arm()


class ThompsonSampling(ActiveBandit):
    """ Thompson Sampling with normally distributed rewards.

        This class cannot be used directly but instead serves as the base class for
        all bandit policies that takes as input multiple active learning rules.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.

        mu : float, optional (default=0.5)
            The initial estimate of the mean of the distribution of the mean reward
            from all arms.

        sigma : float, optional (default=0.02)
            The initial estimate of the variance of the distribution of the mean reward
            from all arms.

        tau : float, optional (default=0.02)
            The initial estimate of the variance of the reward received from all arms.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1, mu=0.5, sigma=0.02, tau=0.02):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.mu = np.full(self.n_arms, mu, dtype=np.float64)
        self.sigma = np.full(self.n_arms, sigma, dtype=np.float64)
        self.tau = np.full(self.n_arms, tau, dtype=np.float64)

        self.mu_history = [self.mu.copy()]
        self.sigma_history = [self.sigma.copy()]
        self.tau_history = [self.tau.copy()]

    def select(self):
        """ Use Thompson sampling to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        # take a sample of rewards from the current prior of heuristics
        sample_rewards = self.seed.normal(self.mu, self.sigma)
        self.selected_arm = np.argmax(sample_rewards)
        return self._select_from_arm()

    def receive_reward(self, reward):
        """ Receive a reward from the environment and updates the policy's prior beliefs.

            Parameters
            ----------
            reward : float
                The reward from the environment should be a good proxy for the decrease
                in the generalisation error of the classifier.
        """
        mu = self.mu[self.selected_arm]
        sigma = self.sigma[self.selected_arm]
        tau = self.tau[self.selected_arm]

        self.mu[self.selected_arm] = (mu * tau + reward * sigma) / (tau + sigma)
        self.sigma[self.selected_arm] = (sigma * tau) / (tau + sigma)
        self.T[self.selected_arm] += 1

        # store results in history
        self.mu_history.append(self.mu.copy())
        self.sigma_history.append(self.sigma.copy())
        self.tau_history.append(self.tau.copy())
        self.T_history.append(self.T.copy())
        self.reward_history.append(reward)

    def history(self):
        """ Return a dictionary containing the history of the policy.

            Returns
            -------
            history : dict
                The dictionary contains the following keys: mu, sigma, T, and reward.
                The corresponding value of each key is an array containing the state
                in each time step.
        """
        history = {}
        history['mu'] = np.array(self.mu_history)
        history['sigma'] = np.array(self.sigma_history)
        history['tau'] = np.array(self.tau_history)
        history['T'] = np.array(self.T_history)
        history['reward'] = np.array(self.reward_history)
        return history


class OCUCB(ActiveBandit):
    """ Optimally Confident UCB (OC-UCB) Policy, based on Lattimore (2015).

        The OC-UCB algorithm is presented in the paper `Optimally Confident UCB
        Improved Regret for Finite-Armed Bandits` by Tor Lattimore in 2015.
        The algorithm is based on UCB and contains two tunable variables,
        ``alpha`` and ``psi``.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.

        alpha : float, optional (default=3)
            Lattimore (2015) found that alpha=3 leads to good results.

        psi : float, optional (default=2)
            Lattimore (2015) found that psi=2 leads to good results.

        horizon : int
            The OC-UCB algorithm requires the knowledge of the horizon, i.e.
            the maximum number of time steps.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1, alpha=3, psi=2,
                 horizon=1000):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.alpha = 3
        self.psi = 2
        self.mu = np.zeros(self.n_arms)
        self.sum_mu = np.zeros(self.n_arms)
        self.mu_history = [self.mu.copy()]
        self.horizon = horizon

    def select(self):
        """ Use the OC-UCB algorithm to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        self.time_step += 1

        if self.time_step <= self.n_arms:
            self.selected_arm = self.time_step - 1
            return self._select_from_arm()

        else:
            ucb = self.mu + np.sqrt((self.alpha / self.T) *
                                     np.log(self.psi * self.horizon / self.time_step))
            self.selected_arm = np.argmax(ucb)
            return self._select_from_arm()


class EXP3PP(ActiveBandit):
    """ EXP3++ policy, as described by Seldin (2014).

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.loss = np.zeros(self.n_arms)
        self.loss_history = [self.loss]

    def select(self):
        """ Use the EXP++ algorithm to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        self.time_step += 1

        # eta is the learning rate
        # xi is the exploration parameter
        beta = 0.5 * np.sqrt(np.log(self.n_arms) / (self.time_step * self.n_arms))
        gap = np.minimum(1, (1 / self.time_step) * (self.loss - np.min(self.loss)))
        xi = 18 * np.log(self.time_step)**2 / (self.time_step * gap**2)
        xi[np.isnan(xi)] = np.inf
        epsilon = np.minimum(1/(2 * self.n_arms), beta)
        epsilon = np.minimum(epsilon, xi)
        eta = beta
        rho = np.exp(-eta * self.loss)
        rho /= np.sum(rho)
        self.rho = (1 - np.sum(epsilon)) * rho + epsilon

        self.selected_arm = self.seed.choice(self.n_arms, p=rho)
        return self._select_from_arm()

    def receive_reward(self, reward):
        """ Receive a reward from the environment and updates the policy's prior beliefs.

            Parameters
            ----------
            reward : float
                The reward from the environment should be a good proxy for the decrease
                in the generalisation error of the classifier.
        """
        loss = 1 - reward
        self.loss[self.selected_arm] += loss / self.rho[self.selected_arm]

        self.T[self.selected_arm] += 1

        # store results in history
        self.T_history.append(self.T.copy())
        self.reward_history.append(reward)
        self.loss_history.append(self.loss.copy())

    def history(self):
        """ Return a dictionary containing the history of the policy.

            Returns
            -------
            history : dict
                The dictionary contains the following keys: mu, sigma, T, and reward.
                The corresponding value of each key is an array containing the state
                in each time step.
        """
        history = {}
        history['T'] = np.array(self.T_history)
        history['reward'] = np.array(self.reward_history)
        history['loss'] = np.array(self.loss_history)
        return history


class KLUCB(ActiveBandit):
    """ kl-UCB policy with normally distributed rewards, as described by Cappé (2013).

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.

        mu : float, optional (default=0)
            The initial estimate of the mean of the distribution of the mean reward
            from all arms.

        sigma : float, optional (default=0.02)
            The initial estimate of the variance of the distribution of reward from all arms.
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1, mu=0, sigma=0.02):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.mu = np.full(self.n_arms, mu, dtype=np.float64)
        self.sum_mu = np.full(self.n_arms, mu, dtype=np.float64)
        self.sigma = sigma
        self.mu_history = [self.mu.copy()]

    def select(self):
        """ Use the kl-UCB algorithm to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        self.time_step += 1

        if self.time_step <= self.n_arms:
            self.selected_arm = self.time_step - 1
            return self._select_from_arm()

        else:
            max_kl = np.log(self.time_step) / self.T
            ucb = self.mu + np.sqrt(2 * self.sigma * max_kl)
            self.selected_arm = np.argmax(ucb)
            return self._select_from_arm()



class COMB(ActiveBandit):
    """ The COMB algorithm described in Baram, El-Yaniv, and Luz (2004).

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.

        alpha : float, optional (default=0)
            The probability threshold

        beta : float, optional (default=0.02)
            The probability scaling factor
    """
    def __init__(self, pool, labels, classifier, arms, random_state=None,
                 n_candidates=None, n_best_candidates=1, alpha=0.05, beta=100):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.alpha = alpha
        self.beta = beta
        self.w = np.ones(self.n_arms, dtype=np.float64)
        self.w_history = [self.w.copy()]
        self.r = 1  # Parameters for reward limit
        assert n_best_candidates == 1, 'COMB only works with 1 candidate at a time.'

    def select(self):
        """ Use the COMB algorithm to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """

        candidate_mask = self._sample()
        n_actual_candidates = np.sum(candidate_mask)
        predictions = self.classifier.predict_proba(self.pool[candidate_mask])

        # Receive advice scoring vectors from active learners
        scores = []
        for arm in self.arms:
            score = arm.score(candidate_mask, predictions)
            # Scale score to be between 0 and 1
            score = (score - score.min()) / (score.max() - score.min())
            scores.append(score)
        scores = np.asarray(scores)

        # Scale advice scoring vectors
        scores = np.exp(-self.beta * (1 - scores))
        Z = np.sum(scores, axis=1)
        Z = np.tile(Z, (n_actual_candidates, 1)).T
        scores = scores / Z

        # Extract effective pool
        candidate_idx = np.where(candidate_mask)[0]
        effective_mask = np.zeros(len(candidate_idx), dtype=bool)
        alpha = self.alpha
        pool_scores = np.full(len(candidate_mask), -np.inf)
        while np.sum(effective_mask) == 0:
            max_scores = np.max(scores, axis=0)
            effective_mask = max_scores >= alpha
            alpha /= 2

        # Keep only scores of the candidates whose score is above threshold
        candidate_idx = candidate_idx[effective_mask]
        candidate_mask = np.zeros(len(candidate_mask), dtype=bool)
        candidate_mask[candidate_idx] = True
        new_scores = []
        for score in scores:
            new_scores.append(score[effective_mask])
        new_scores = np.asarray(new_scores)

        n_eff_candidates = np.sum(effective_mask)
        g_max = (n_eff_candidates * np.log(self.n_arms) / np.exp(1) - 1) * 4**self.r
        gamma = np.sqrt((n_eff_candidates * np.log(self.n_arms)) /
                        ((np.exp(1) - 1) * g_max))

        W = np.sum(self.w)
        w = np.tile(self.w, (n_eff_candidates, 1)).T
        p = (1 - gamma) * np.sum(w * new_scores) / W + gamma / n_eff_candidates

        # Sort from largest to smallest and pick the candidate(s) with the highest score(s)
        best_candidates = np.argsort(-p)[:1]
        self.p_max = np.max(p)
        self.p_max_idx = np.argmax(p)

        self.scores = new_scores
        self.n_eff_candidates = n_eff_candidates
        self.gamma = gamma

        return best_candidates

    def receive_reward(self, reward):
        """ Receive a reward from the environment and updates the policy's prior beliefs.

            Parameters
            ----------
            reward : float
                The reward should be the classification entropy maximization
                score.
        """
        r_hat = reward / self.p_max
        for i in range(self.n_arms):
            self.w[i] = self.w[i] * np.exp(self.scores[i][self.p_max_idx] * self.gamma / self.n_eff_candidates)

        self.w_history.append(self.w.copy())


class ActiveAggregator(MultipleSuggestions):
    """ Aggregator policies that takes multiple active learning rules.

        The ranks from the arms can be combined using the Borda count, the geometric
        mean, or the Schulze method.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        classifier : Classifier object
            The classifier should have the same interface as scikit-learn classifier.
            In particular, it needs to have the fit and predict methods.

        arms : array of Arm objects
            Each arm is a particular active learning rule. The arm needs to implement
            the ``select`` method that returns an array of indices of objects from the
            pool for labelling.

        aggregator : {'borda', 'geometric', 'schulze'}, optional (default='borda')
            The type of aggregator function used to combine the ranks from the arms.
            The arithmetic mean (Borda count) and the geometric mean are the
            most efficient algorithms. The Schluze method is recommended only when the
            candidate set is small, since the algorithm uses pairwise counting and runs
            in ``O(n^3)`` where ``n`` is the number of candidates.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        n_candidates : int, optional (default=None)
            The number of candidates in the unlabelled pool to be chosen for evaluation
            at each iteration. For very large datasets, it might be useful to limit
            the the number of candidates to a small number (like 300) since some
            policies can take a long time to run. If not set, the whole unlabelled
            pool will be used.

        n_best_candidates : int, optional (default=1)
            The number of candidates returned at each iteration for labelling. Batch-mode
            active learning is where this parameter is greater than 1.
    """
    def __init__(self, pool, labels, classifier, arms, aggregator='borda',
                 random_state=None, n_candidates=None, n_best_candidates=1):
        super().__init__(pool, labels, classifier, arms, random_state,
                         n_candidates, n_best_candidates)
        self.aggregator = aggregator

        if aggregator == 'borda':
            self._aggregate_votes = self._aggregate_votes_borda
        elif aggregator == 'geometric':
            self._aggregate_votes = self._aggregate_votes_geometric
        elif aggregator == 'schulze':
            self._aggregate_votes = lambda voters: _aggregate_votes_schulze(
                                    voters, self.n_best_candidates)
        else:
            raise ValueError("The aggregator argument must be one of " +
                             "{'borda', 'geometric', 'schulze'}")

    def select(self):
        """ Use the chosen aggregation method to choose the next candidates for labelling.

            Returns
            -------
            best_candidates : array of ints
                An array of indices of objects in the pool.
        """
        candidate_mask = self._sample()
        predictions = self.classifier.predict_proba(self.pool[candidate_mask])

        voters = [arm.select(candidate_mask, predictions, self.n_candidates) for arm in self.arms]
        voters = np.array(voters)
        best_candidates = self._aggregate_votes(voters)
        return best_candidates

    def _aggregate_votes_borda(self, voters):
        """ Aggregate the ranks from the arms using Borda count. """
        max_score = len(voters[0])
        points = {}

        # accumulate the points for each candidate
        for voter in voters:
            for idx, candidate in enumerate(voter):
                points[candidate] = points.get(candidate, 0) + max_score - idx

        # sort the candidates and return the most popular one(s)
        rank = sorted(points, key=points.__getitem__, reverse=True)
        return rank[:self.n_best_candidates]

    def _aggregate_votes_geometric(self, voters):
        """ Aggregate the ranks from the arms using the geometric mean. """
        max_score = len(voters[0])
        points = {}

        # accumulate the points for each candidate
        for voter in voters:
            for idx, candidate in enumerate(voter):
                points[candidate] = points.get(candidate, 1) * (max_score - idx)

        # sort the candidates and return the most popular one(s)
        rank = sorted(points, key=points.__getitem__, reverse=True)
        return rank[:self.n_best_candidates]
