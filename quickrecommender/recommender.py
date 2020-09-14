import numpy as np
from .utils.diversifier import kmeanspp
from .utils.ops import cosine_similarity, softmax
from .graphs.nearestneighbors import NearestNeighbors


class QuickRecommender:
    """
    QuickRecommender
    Creates a content-based model using a nearest-neighbors graph, updates user-preferences
    based on the graph, diversifies using K-Means++, and returns the results.
    Each user will have a `user-vector` which is basically an array with length N, where N
    is the number of samples.
    The recommender requires the parameters `n_neighbors`, and `metric`.
     `n_neighbors` specifies the number of nearest neighbors to store.
     `metric` specifies the similarity metric, which can be set to any callable that inputs
        a matrix and returns its self-similarity NxN matrix. It can also be set to the string
        ``cosine`` in order to use cosine similarity.
    """
    def __init__(self, n_neighbors=50, metric='cosine'):
        self._input_data = None
        self._n_samples = -1
        self._nn_graph = None
        self._n_neighbors = n_neighbors

        if type(metric) is callable:
            self.similarity_fn = metric
        elif type(metric) is str:
            if metric == 'cosine':
                self.similarity_fn = cosine_similarity
            else:
                raise RuntimeError('Metric `{}` not supported.'.format(metric))
        else:
            raise RuntimeError('Metric of type {} not supported.'.format(type(metric)))

    def fit(self, input_data):
        """
        Creates the nearest-neighbors graph and stores it
        """
        self._input_data = input_data
        self._n_samples = input_data.shape[0]
        self._nn_graph = NearestNeighbors(n_samples=self._n_samples, n_neighbors=self._n_neighbors)
        self._nn_graph.fit(input_data, similarity_fn=self.similarity_fn)

    def recommend(self, user_vector=None, n_recommendations=10):
        """
        Recommends a given user items based on its preferences or `user_vector`
        :param user_vector: ndarray, or list of size n_samples or None (initial state)
        :param n_recommendations:
        :return: list of recommended items` indices
        """
        assert n_recommendations <= self._n_samples
        assert self._nn_graph is not None
        if user_vector is None:
            user_vector = np.ones(self._n_samples, dtype=float)
        assert len(user_vector) == self._n_samples
        if type(user_vector) is list:
            user_vector = np.array(user_vector)

        n_init = min(n_recommendations * 3, self._n_samples)

        initial_recommendations_idx = np.random.choice(self._n_samples, n_init, replace=False, p=softmax(user_vector))
        diversified_idx = kmeanspp(self._input_data[initial_recommendations_idx, :], n_recommendations)

        return initial_recommendations_idx[diversified_idx]

    def update(self, user_vector=None, selections=None):
        """
        Updates the user vector based on the user's selections (indices of samples)
        :param user_vector: ndarray, or list of size n_samples or None (initial state)
        :param selections: list of selected items` indices
        :return: updated user vector
        """
        assert self._nn_graph is not None
        if selections is None:
            return None
        if user_vector is None:
            user_vector = np.zeros(self._n_samples, dtype=float)
        assert len(user_vector) == self._n_samples
        if type(user_vector) is list:
            user_vector = np.array(user_vector)

        for s in selections:
            neighbors = list(self._nn_graph.neighbors[s, :])
            user_vector[neighbors] = np.max([user_vector[neighbors], self._nn_graph.similarities[s, :]], axis=0)

        return user_vector
