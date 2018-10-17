import numpy as np
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class Clustering:
    def __init__(self):
        pass


class Kmeans(Clustering):
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        super().__init__()
        self.cluster = KMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state,
                              copy_x, n_jobs, algorithm)
