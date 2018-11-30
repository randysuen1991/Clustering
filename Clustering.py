import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self):
        self.x_train = None
        self.standardization = False
        self.cluster = None
        self.transform_x_train = None
        self.index_x_train = None
        self.mean = None
        self.std = None


class Kmeans(Clustering):
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        super().__init__()
        self.cluster = KMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state,
                              copy_x, n_jobs, algorithm)

    def standardize(self, x):
        if self.mean is None or self.std is None:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / self.std

    # use elbow method to determine the k.
    @classmethod
    def kmeans_wok(cls, x_train, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                   verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        distortions = []
        k_cand = range(1, 21)
        for k in k_cand:
            kmeanmodel = KMeans(n_clusters=k).fit(x_train)
            distortions.append(sum(np.min(cdist(x_train, kmeanmodel.cluster_centers_, 'euclidean'), axis=1)) /
                               x_train.shape[0])

        plt.plot(distortions)
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        print('Please input the number of clusters:')
        n_clusters = int(input())
        return cls(n_clusters, init, n_init, max_iter, tol, precompute_distances,
                   verbose, random_state, copy_x, n_jobs, algorithm)

    def fit(self, x_train, standardize=False):
        if standardize:
            self.standardization = standardize
            x_train = self.standardize(x_train)
        self.x_train = x_train
        self.cluster.fit(self.x_train)
        self.transform_x_train = self.cluster.transform(x_train)
        self.index_x_train = self.cluster.predict(x_train)

    def predict(self, x_test):
        if self.standardization:
            x_test = self.standardize(x_test)
        return self.cluster.predict(x_test)

