# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:58:45 2021

@author: XHK
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator

demo_data = pd.read_pickle("./demo_data.pkl")
X_vars_used = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
dem_data = demo_data[X_vars_used].values
# #############################################################################
# Generate sample data

from sklearn.cluster import KMeans

X = StandardScaler().fit_transform(dem_data)

kmeans = KMeans(
      init="random",
       n_clusters=16,      # 16 found to be best based on SSE elbow
      n_init=10,
     max_iter=300,
      random_state=42)

kmeans.fit(dem_data)

def get_cluster_labels_and_centres():
    return kmeans.labels_, kmeans.cluster_centers_
# ============================================================================= 
  #              For finding optimal number of clusters: (result = 16)
      
# kmeans_kwargs = {
#         "init": "random",
#        "n_init": 10,
#         "max_iter": 300,
#         "random_state": 42,
#     }
#    
#     # A list holds the SSE values for each k
# sse = []
# 
# maxx = 50
# =============================================================================
# =============================================================================
# 
#
# for k in range(1, maxx):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(dem_data)
#     sse.append(kmeans.inertia_)
#     
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, maxx), sse)
# plt.xticks(range(1, maxx))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
# 
# kl = KneeLocator(
# range(1, maxx), sse, curve="convex", direction="decreasing"
# )
# 
# elbow = kl.elbow
# print('elbow = ', elbow)
# print('thus optimal cluster size = ', elbow)
# =============================================================================

print("TSNE: ")
from sklearn.manifold import TSNE
X = demo_data[X_vars_used].values
X_embedded = TSNE(n_components=3).fit_transform(X)
X_embedded.shape
print(X_embedded)
print(X_embedded.shape)

def get_TSNE_embeddings(n = 3):
    print('getting TSNE embeddings of length: ' +str(n))
    from sklearn.manifold import TSNE
    X = demo_data[X_vars_used].values
    X_embedded= TSNE(n_components=n).fit_transform(X)
    X_embedded.shape
    print(X_embedded)
    print(X_embedded.shape)
    return X_embedded

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2])