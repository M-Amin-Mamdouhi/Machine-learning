import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
"""DBSCAN"""
mnist_train=pd.read_csv('mnist_train.csv')
mnist_test=pd.read_csv('mnist_test.csv')
results = mnist_train['label'] 
predictors = mnist_train.drop(['label'], axis = 1)
X = predictors.to_numpy()
Y = results.to_numpy()
X_std = X/255
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_std)
plt.plot(X_2d[:, 0], X_2d[:, 1], 'k.')
clu = DBSCAN(eps=0.1, min_samples=8, n_jobs=-1)
labels = clu.fit_predict(X_2d)
y = clu.labels_
print('number of clusters= ', y.max())
plt.plot(X_2d[y == -1, 0], X_2d[y == -1, 1], 'k.')
for k in range(0,y.max()+1):
    plt.plot(X_2d[y == k, 0], X_2d[y == k, 1], '.')
plt.show()
n_noise= list(labels).count(-1)
print('number of noise= ', n_noise)
for i in range(-1, y.max()+1):
    X_re= pca.inverse_transform(X_2d[y==i])
    print("X's whith label=", i, 'are:')
    print(X_re)
"""K-Means"""
clu = KMeans(n_clusters=10, n_jobs=-1)
clu.fit_predict(X_2d)
centers = clu.cluster_centers_
y = clu.labels_
print(clu.inertia_/clu.n_clusters)
colors = 'rgbycmrgby'
for k in range(clu.n_clusters):
    plt.plot(X_2d[y == k, 0], X_2d[y == k, 1], colors[k]+'.')
    plt.plot(clu.cluster_centers_[k, 0], clu.cluster_centers_[k, 1], 'kx', markersize=12)
plt.show()
"""Accuracy"""
inertias = []

for k in range(1,15):
    clu = KMeans(n_clusters=k)
    clu.fit_predict(X_2d)
    inertias.append(clu.inertia_)

plt.figure(figsize=(15, 6))    
plt.plot(range(1,15), inertias, 'bo-')
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.show()
totalAccuracy = 0
n_clusters=10
for k in range(1, n_clusters+1):
    clusterAccuracy = max(y==k) / sum(y==k) * 100.0
    totalAccuracy += clusterAccuracy
print("KMeans clustering Accuracy " + str(totalAccuracy / 10))
