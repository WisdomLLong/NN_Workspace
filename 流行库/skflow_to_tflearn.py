# -*- coding: utf-8 -*-

from sklearn import datasets, metrics, preprocessing, cross_validation
import numpy as np

boston = datasets.load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                            test_size = 0.25, random_state =33)
# 对数据特征进行标准化处理
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#@@ 据说skflow已经被集成到TensorFlow的learn中了，所以有下面的写法
#@@ 但是下面的这个仍然存在问题
import tensorflow.contrib.learn.python.learn as learn
import tensorflow as tf
# 使用skflow的LinearRegressor
tf_lr = learn.LinearRegressor(feature_columns=learn.infer_real_valued_columns_from_input(X_train), \
                optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))
#@@ optimizer是设置优化器的，默认的梯度0.2会发生梯度爆炸

tf_lr.fit(X_train, y_train, steps=1000, batch_size=50)
tf_lr_y_predict = tf_lr.predict(X_test)
tf_lr_y_predict = np.array(list(tf_lr_y_predict))

print('absoluate error:', metrics.mean_absolute_error(tf_lr_y_predict, y_test),'\n')
print('mean squared error:', metrics.mean_squared_error(tf_lr_y_predict, y_test),'\n')
print('R-squared value:', metrics.r2_score(tf_lr_y_predict, y_test))