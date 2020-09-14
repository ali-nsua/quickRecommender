import numpy as np


class NearestNeighbors:
    def __init__(self, n_samples, n_neighbors=50):
        assert n_samples >= n_neighbors
        self.neighbors = np.zeros((n_samples, n_neighbors), dtype=int)
        self.similarities = np.zeros((n_samples, n_neighbors), dtype=float)

    def fit(self, input_data, similarity_fn):
        n_samples, n_neighbors = self.similarities.shape
        assert n_samples == input_data.shape[0]
        sorted_idx_end = n_samples - n_neighbors - 1

        similarities = similarity_fn(input_data)
        self.neighbors = np.argsort(similarities, axis=1, kind='mergesort')[:, n_samples:sorted_idx_end:-1]

        rows = [i for i in range(n_samples) for j in range(n_neighbors)]
        cols = self.neighbors.ravel()

        self.similarities = similarities[rows, cols]
