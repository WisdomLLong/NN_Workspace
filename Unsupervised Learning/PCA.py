# -*- coding: utf-8 -*-
# 降维
# 对手写体数字图形进行降维

import pandas as pd
import numpy as np
# 训练集
digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
# 测试集
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# print(digits_train.head())

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

from sklearn.decomposition import PCA

# 初始化PCA，并进行压缩
extimator = PCA(n_components=2)
X_pca = extimator.fit_transform(X_digits)

# 显示2维分布
# 这里使用了pandas自带的，准确说是DataFrame类的plot类
X_plot_test = pd.DataFrame(X_pca,columns=['w','h'])
X_plot_test.plot.scatter(x='w', y='h')

from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors = ['black','blue','purple','yellow','white','red','lime',\
              'cyan', 'orange', 'gray']
              
    for i in range(len(colors)):
        # y_digits.as_matrix()是转变为Numpy-array格式
        # 后面的==i会返回一个bool类型的array
        # array类型的X_pca的后面可以筛选掉不需要的数据
        px = X_pca[:, 0][y_digits.as_matrix()==i]
        py = X_pca[:, 1][y_digits.as_matrix()==i]
        plt.scatter(px, py, c=colors[i])
        
    plt.legend(range(10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
plot_pca_scatter()










