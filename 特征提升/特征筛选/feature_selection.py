# -*- coding: utf-8 -*-
# 任然是对Titanic数据进行决策树分类
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())

X = titanic.drop(['row.names', 'name', 'survived'], axis = 1)
#@@ df.mean(axis=1)是指“矩阵水平方向”上的均值
#@@ df.drop("col4", axis=1)是指删除在“水平方向”上的列名为“col4”的列
y = titanic['survived']

#@@ inplace = True 是指修改元数据但不产生副本
X['age'].fillna(X['age'].mean(), inplace = True)
X.fillna('UNKNOWN', inplace = True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# 类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
"""
  (样本标号，类型)   值
print(X_train)
  (0, 0)        31.1941810427
  (0, 78)       1.0
  (0, 82)       1.0
  (0, 366)      1.0
  (0, 391)      1.0
  (0, 435)      1.0
  (0, 437)      1.0
  (0, 473)      1.0
  (1, 0)        31.1941810427
  (1, 73)       1.0
  (1, 79)       1.0
  (1, 296)      1.0
  (1, 389)      1.0
  (1, 397)      1.0
  (1, 436)      1.0
  (1, 446)      1.0
  :     :
  (983, 0)      31.1941810427
  (983, 78)     1.0
  (983, 82)     1.0
  (983, 366)    1.0
  (983, 391)    1.0
  (983, 435)    1.0
  (983, 436)    1.0
  (983, 473)    1.0
"""
# 输出处理后特征向量的维度
print(len(vec.feature_names_))


# 使用决策树模型依靠所有特征进行预测，并作性能估计
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt.score(X_test, y_test)



#------------------------------------------------------------------------------
# 特征筛选
#------------------------------------------------------------------------------
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)


#------------------------------------------------------------------------------
# 通过交叉验证的方法，按照固定间隔的百分比筛选特征
#------------------------------------------------------------------------------
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv = 5)
    results = np.append(results, scores.mean())
print(results)

# 找到提现最佳性能的特征筛选的百分比
opt = np.where(results == results.max())[0]
print('Optimal number of feature %d' %percentiles[opt])

import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,\
                        percentile=percentiles[opt])
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)















