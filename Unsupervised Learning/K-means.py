# -*- coding: utf-8 -*-
# 数学运算，做图，数据分析
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
#@@ header=None时，即指明原始文件数据没有列索引
#@@ 这样read_csv为自动加上列索引，除非你给定列索引的名字。
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

# 从训练和测试数据集上都分离出64维度的像素attribute与1维度的数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]                     


#------------------------------------------------------------------------------
# 训练
#------------------------------------------------------------------------------
from sklearn.cluster import KMeans

km = KMeans(n_clusters=10)
km.fit(X_train)
km_y_predict = km.predict(X_test)

#------------------------------------------------------------------------------
# 两种评价机制：ARI和轮廓系数
#------------------------------------------------------------------------------
# ARI
from sklearn import metrics
print(metrics.adjusted_rand_score(y_test, km_y_predict))

# 轮廓系数
from sklearn.metrics import silhouette_score
#@@ 分割出3*2=6个子图，并在1号子图作图
plt.subplot(3,2,1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
#@@ 注意3.x版本的Python，Zip返回的是一个迭代器，而不是一个list

# 在1号子图做出原始数据点阵的分布
plt.title('Instances')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model =  KMeans(n_clusters=t).fit(X)
    what = 'dd'
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color = colors[l], marker = markers[l], ls = 'None')
    
    plt.xlim([0 ,10])
    plt.ylim([0, 10])
    
    # 计算轮廓系数
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coefficient=%0.03f'%(t, sc_score))
    
# 绘制轮廓系数与类簇数量的关系图

plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()
    









