# -*- coding: utf-8 -*-
# 美国波士顿地区房价数据描述

# 读取数据
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)

# 数据分隔
from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, \
                    test_size = 0.25)
'''
print("The max target value is", np.max(y))
print("The min target value is", np.min(y))
print("The average target value is", np.mean(y))
'''

#------------------------------------------------------------------------------
# 数据标准化处理
#------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
# 分别初始化对特征和目标值的标准化
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)


#------------------------------------------------------------------------------
# 两种配置的k-means：平均回归，距离加权回归
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsRegressor

# 平均回归
uni_knr = KNeighborsRegressor(weights= 'uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 距离加权回归
dis_knr = KNeighborsRegressor(weights= 'distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

#------------------------------------------------------------------------------
# 三种方法进行性能预测，这里不再赘述，与线性回归中的相同
#------------------------------------------------------------------------------












