# -*- coding: utf-8 -*-
#@@ 通过“肘部”观察法确定最合适的聚类中心个数

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

plt.subplot(2, 1, 1)
# 绘制30个数据样本点的分布图像
Y = np.hstack((cluster1, cluster2, cluster3))
X = np.hstack((cluster1, cluster2, cluster3)).T
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')

# 测试9种不同聚类中心数量下，每种聚类情况的聚类质量，并作图
K = range(1,10)
mean_distortions = []

for k in K:
    kmeans = KMeans(n_clusters= k)
    kmeans.fit(X)
    # 这里的min是对聚类中心来说的，sum是对30个数据点来说的
    mean_distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,'euclidean'),\
                            axis=1))/X.shape[0])
    
plt.subplot(2, 1, 2)
plt.plot(K, mean_distortions, 'rx-')
plt.xlabel('聚类中心个数k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()

