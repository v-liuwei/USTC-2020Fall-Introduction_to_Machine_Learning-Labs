import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import Literal


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def DBI(clusters):
    k = len(clusters)
    p = clusters[0].shape[1]
    centroids = np.zeros((k, p))
    avgs = np.zeros(k)
    d = np.zeros((k, k))
    valid_clusters = clusters.copy()
    for c in clusters:
        if clusters[c].size == 0:
            valid_clusters.pop(c)
    clusters = valid_clusters.copy()
    for c in clusters:
        ctrd = np.mean(clusters[c], axis=0)
        centroids[c] = ctrd
        avgs[c] = np.mean([dist(x, ctrd) for x in clusters[c]])
    for ci in clusters:
        for cj in clusters:
            d[ci, cj] = dist(centroids[ci], centroids[cj])
    return np.mean([np.max([(avgs[ci] + avgs[cj]) / d[ci, cj] for cj in clusters if cj != ci]) for ci in clusters])


class KMeans(object):
    def __init__(self,
                 n_clusters: int,
                 *,
                 init: Literal['random', 'from_data'] = 'random',
                 stop: Literal['centroids', 'labels'] = 'centroids',
                 n_init: int = 10,
                 max_iter: int = 300,
                 random_state: int = None):
        """
        n_clusters: number of clusters
        init: method for centroids initialization
            - 'random': generate randomly
            - 'from_data': select from data randomly
        stop: criteria of algorithm termination
            - 'centroids': when centroids do not change
            - 'labels': when labels do not change
        n_init: number of time the k-means algorithm will be run with different centroid seeds.
        max_iter: maximum number of iterations of the k-means algorithm for a single run.
        random_state: seed of random number generator
        """
        self.__k = n_clusters
        self.__init = init
        self.__stop = stop
        self.__n_init = n_init
        self.__max_iter = max_iter
        self.__seed = random_state

    def fit(self, X: np.ndarray):
        self.__data = X
        seeds = list(np.random.randint(0, 65535, size=self.__n_init))
        seeds[0] = self.__seed
        min_dbi = np.inf
        for _i in range(self.__n_init):
            n_iter, centroids, labels = self.__run(seeds[_i])
            clusters = dict()
            for c in range(self.__k):
                clusters[c] = self.__data[labels == c]
            dbi = DBI(clusters)
            if dbi < min_dbi:
                self.centroids_ = centroids
                self.labels_ = labels
                self.clusters_ = clusters
                self.n_iter_ = n_iter
        return self

    def plot_clusters(self):
        fig, ax = plt.subplots()
        ax.set_title("Clusters of input data")
        ax.set_xlabel(r"$X_{1}$")
        ax.set_ylabel(r"$X_{2}$")
        ax.scatter(self.__data[:, 0], self.__data[:, 1], s=20, c=self.labels_)
        classes = np.unique(self.labels_)
        ax.scatter(self.centroids_[classes, 0], self.centroids_[classes, 1],
                   s=200, c=classes, marker='X')
        plt.show()

    def __run(self, seed):
        centroids = self.__gen_centroids(seed)
        labels = - np.ones(self.__data.shape[0], dtype=int)
        for i in range(self.__max_iter):
            new_labels = self.__update_labels(centroids)
            new_centroids = self.__update_centroids(centroids, new_labels)
            terminate = False
            if self.__stop == 'centroids' and (new_centroids == centroids).all():
                terminate = True
            elif self.__stop == 'labels' and (new_labels == labels).all():
                terminate = True
            centroids = new_centroids
            labels = new_labels
            if terminate:
                break
        n_iter = i
        return n_iter, centroids, labels

    def __gen_centroids(self, seed):
        k, (m, p) = self.__k, self.__data.shape
        np.random.seed(seed)
        if self.__init == 'random':
            centroids = np.zeros((k, p))
            for j in range(p):
                dmin, dmax = np.min(self.__data[:, j]), np.max(self.__data[:, j])
                centroids[:, j] = np.random.rand(k) * (dmax - dmin) + dmin
            return centroids
        elif self.__init == 'from_data':
            indices = np.random.choice(m, k)
            return self.__data[indices]

    def __update_labels(self, centroids):
        m = self.__data.shape[0]
        labels = np.zeros(m, dtype=int)
        for j in range(m):
            dists = np.array([dist(self.__data[j], ctrd) for ctrd in centroids])
            labels[j] = np.argmin(dists)
        return labels

    def __update_centroids(self, old_centroids, labels):
        centroids = old_centroids.copy()
        for _k in range(self.__k):
            cluster = self.__data[labels == _k]
            if cluster.size > 0:
                centroids[_k] = np.mean(cluster, axis=0)
        return centroids
