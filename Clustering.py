import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.standardization = False
        self.cluster = None
        self.transform_x_train = None
        self.index_x_train = None


class Kmeans(Clustering):
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        super().__init__()
        self.cluster = KMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state,
                              copy_x, n_jobs, algorithm)

    @staticmethod
    def standardize(x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / std

    # use elbow method to determine the k.
    @classmethod
    def kmeans_wok(cls, x, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                   verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        distortions = []
        k_cand = range(1, 10)
        for k in k_cand:
            kmeanmodel = KMeans(n_clusters=k).fit(x)
            kmeanmodel.fit(x)
            distortions.append(sum(np.min(cdist(x, kmeanmodel.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])

        plt.plot(distortions)
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        n_clusters = input()
        return cls(n_clusters, init, n_init, max_iter, tol, precompute_distances,
                   verbose, random_state, copy_x, n_jobs, algorithm)

    def fit(self, x_train, y_train, standardize=False):
        if standardize:
            self.standardization = standardize
            x_train = self.standardize(x_train)
        self.x_train = x_train
        self.y_train = y_train
        self.cluster.fit(x_train)
        self.transform_x_train = self.cluster.transform(x_train)
        self.index_x_train = self.cluster.predict(x_train)

    def predict(self, x_test):
        return self.cluster.predict(x_test)

