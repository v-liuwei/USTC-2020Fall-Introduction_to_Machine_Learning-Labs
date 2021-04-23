import numpy as np
from kmeans import KMeans, DBI

# load data
data = np.array(list(np.load('./k-means/k-means.npy').item().values()))
data = np.array(data).reshape((-1, data.shape[2]))

# create a kmeans model and fit it
model = KMeans(n_clusters=3, init='random', stop='centroids', n_init=5, max_iter=30, random_state=123)
model.fit(data)

# output centroids and calculate DBI
print('cluster centroids are:')
print(model.centroids_)
print("DBI = {}".format(DBI(model.clusters_)))

# plot clusters
model.plot_clusters()
