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
print("The max target value is", np.max(y))
print("The min target value is", np.min(y))
print("The average target value is", np.mean(y))

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
# 线性回归和SGD回归
#------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)

#------------------------------------------------------------------------------
# 三种方法进行性能预测
#------------------------------------------------------------------------------
# 使用LinearRegression模型自带的评估模块
print('The value of default measurement of LinearRegression is', \
      lr.score(X_test, y_test))
#@@ 注意这里没有使用lr_y_predict，是因为score函数里面又计算了一遍self.predict(X)


# 三个回归评价机制
# r2_score, mean_squared_error, mean_absoluate_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print('The value of R-squared of LinearRegression is', r2_score(y_test,lr_y_predict))

print('The mean squared error of LinearRegression is', mean_squared_error(\
     ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
#@@ inverse_transform是对transform的反转
print('The mean absoluate error of LinearRegression is', mean_absolute_error(\
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))


# 使用SGDRegression模型自带的评估模块
print('The value of default measurement of SGDRegression is', sgdr.score(X_test, y_test))

print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))

print('The mean squrard error of SGDRegressor is', mean_squared_error(\
     ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))

print('The mean absoluate error of LinearRegression is', mean_absolute_error(\
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))















