"""
MinMax k-Means
*************

:Description: MinMax k-Means (Tzortzis, G. & Likas, A. The MinMax k-Means clustering algorithm
              Pattern Recognition , 2014, 47, 2505 - 2516 1) implementation following Sklearn API conventions.
:Authors: Kevin Rosales
:Version: 1.0
:Created on: 16/05/2021 19:50
"""

__author__ = 'Kevin Rosales'

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import euclidean_distances


def ssd(X, Y):
    return np.square(euclidean_distances(X, Y))


class MinMaxKMeans(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, n_clusters, p_max: float = 0.5, p_step: float = 0.01,
                 beta: float = 0.1, epsilon: float = 10E-6, t_max: int = 500,
                 random_state: int = None):
        """
        Initialization of the MinMaxKMeans Object

        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param p_max: Maximum value that 'p' can attain.
        :param p_step: Change step in 'p' value.
        :param beta: Influence of the previous iteration weights to the current update.
        :param epsilon: Termination criterion that measures the change of the relaxed objective's value between two
        consecutive iterations
        :param t_max: Maximum number of iterations.
        :param random_state: Determines random number generation for centroid initialization. Use an int to make the
        randomness deterministic.
        """
        self.n_clusters = n_clusters
        self.p_max = p_max
        self.p_step = p_step
        self.beta = beta
        self.epsilon = epsilon
        self.t_max = t_max
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.w = None
        self.p = None

    def fit(self, X, y=None, sample_weight=None):
        """
        Compute MinMax k-means clustering.

        :param X: Training instances to cluster.
        :param y: Not used, present here for API consistency by convention.
        :param sample_weight: Not used, present here for API consistency by convention.
        """
        rs = np.random.RandomState(seed=self.random_state)
        p_init = 0
        self.p = p_init
        self.w = [1/self.n_clusters for _ in range(self.n_clusters)]
        random_idxs = rs.randint(X.shape[0], size=self.n_clusters)
        self.cluster_centers_ = [X[i] for i in random_idxs]
        self.labels_ = np.argmin(np.multiply(np.power(self.w, self.p), ssd(X, self.cluster_centers_)), axis=1)
        empty = False
        last_E = self._get_Ew(X)
        stored_assign = {}
        stored_weights = {}
        for self.n_iter_ in range(1, self.t_max+1):
            self.labels_ = np.argmin(np.multiply(np.power(self.w, self.p), ssd(X, self.cluster_centers_)), axis=1)
            if np.min([np.where(self.labels_ == i)[0].shape[0] for i in range(self.n_clusters)]) < 2:
                empty = True
                self.p = round(self.p - self.p_step, 2)
                if self.p < p_init:
                    return
                self.labels_ = stored_assign[self.p]
                self.w = stored_weights[self.p]

            self.cluster_centers_ = [np.sum(X[np.where(self.labels_ == k)], axis=0) /
                                     np.where(self.labels_ == k)[0].shape[0]
                                     for k in range(self.n_clusters)]

            if self.p < self.p_max and empty is False:
                stored_assign[self.p] = self.labels_.copy()
                stored_weights[self.p] = self.w
                self.p = round(self.p + self.p_step, 2)

            v = [np.sum(ssd(X[np.where(self.labels_ == k)],
                            [self.cluster_centers_[k]]).reshape(-1))
                 for k in range(self.n_clusters)]

            self.w = [self.beta * self.w[k] + (1 - self.beta) * (np.power(v[k], 1/(1 - self.p)) /
                                                                 np.sum(np.power(v, 1/(1 - self.p))))
                 for k in range(self.n_clusters)]

            new_E = self._get_Ew(X)
            if np.abs(new_E - last_E) < self.epsilon:
                break
            last_E = new_E

        self.inertia_ = np.sum(v)

    def _get_Ew(self, X) -> float:
        """
        Computes the relaxed maximum variance objective.

        :param X: Training instances to cluster.
        :return: Relaxed maximum variance objective
        """
        return np.sum([np.power(self.w[k], self.p)*np.sum(ssd(X[np.where(self.labels_ == k)],
                                                          [self.cluster_centers_[k]]).reshape(-1))
                       if np.where(self.labels_ == k)[0].shape[0] > 0 else 0
                       for k in range(self.n_clusters)])

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        :param X: New data to transform.
        :param y: Not used, present here for API consistency by convention.
        :param sample_weight: Not used, present here for API consistency by convention.
        :return: labels: Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X).

        :param X: New data to transform.
        :param y: Not used, present here for API consistency by convention.
        :param sample_weight: Not used, present here for API consistency by convention.
        :return: X transformed in the new space.
        """
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True) -> dict:
        """
        Get parameters for this estimator.

        :param deep: Not used, present here for API consistency by convention.
        :return: Parameter names mapped to their values.
        """
        return {
            'beta': self.beta,
            'epsilon': self.epsilon,
            'n_clusters': self.n_clusters,
            'p_max': self.p_max,
            'p_step': self.p_step,
            'random_state': self.random_state,
            't_max': self.t_max
        }

    def predict(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: New data to predict.
        :param sample_weight: Not used, present here for API consistency by convention.
        :return: Index of the cluster each sample belongs to.
        """
        return np.argmin(np.multiply(np.power(self.w, self.p), ssd(X, self.cluster_centers_)), axis=1)

    def score(self, X, y=None, sample_weight=None) -> float:
        """
        Opposite of the value of X on the MinMax K-means objective.

        :param X: New data.
        :param y: Not used, present here for API consistency by convention.
        :param sample_weight: Not used, present here for API consistency by convention.
        :return: Opposite of the value of X on the MinMax K-means objective (i.e. relaxed maximum variance objective).
        """
        return -self._get_E(X)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param params: Estimator parameters.
        """
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.

        :param X: New data to transform.
        :return: X transformed in the new space.
        """
        return ssd(X, self.cluster_centers_)

