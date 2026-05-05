from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

data=load_iris()
X=data.data[:,:2]

km=KMeans(n_clusters=3,random_state=42,n_init=10)
km.fit(X)
labels=km.labels_
centers=km.cluster_centers_

colors=['r','g','b']
for i in range(3):
    pts=X[labels==i]
    plt.scatter(pts[:,0],pts[:,1],c=colors[i],label=f'Cluster {i+1}')
plt.scatter(centers[:,0],centers[:,1],c='k',marker='X',s=200,label='Centroids')
plt.title('K-Means Clustering (Iris Dataset)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.tight_layout()
plt.show()
print("Cluster Centers:\n",centers)
print("Labels:",labels)
