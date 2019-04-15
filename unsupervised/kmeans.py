import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import pylab as pl



##Generate Sample data
centers = [[1, 1], [-1,1], [-1,1], [1, -1]]
X, labels_true= make_blobs(n_samples=750, centers = centers, cluster_std=0.4)

#print(type(X))   #<class 'numpy.ndarray'>
#print(type(labels_true))  #<class 'numpy.ndarray'>

kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
print(kmeans)   #KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,     n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',     random_state=None, tol=0.0001, verbose=0)

kmeans.fit(X)    #KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,     n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',     random_state=None, tol=0.0001, verbose=0)

centroids = kmeans.cluster_centers_
'''print(centroids)
array([[-0.82420814,  0.63130234],
       [ 1.01869188, -0.95397485],
       [ 0.97822737,  1.01222076],
       [-1.08981052,  1.25686097]])
type(centroids)   #<class 'numpy.ndarray'>'''

labels = kmeans.labels_

#type(labels)  #<class 'numpy.ndarray'>



