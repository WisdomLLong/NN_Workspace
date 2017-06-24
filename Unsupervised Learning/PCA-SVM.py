# -*- coding: utf-8 -*-
# 降维
# 对手写体数字图形进行降维

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
# 训练集
digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
# 测试集
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)


# X_train = digits_train[list(range(64))]
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

##########################################
# 不使用pca的svm
#########################################
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

print(svc.score(X_test, y_test))
# target_names表示分类类别的名字
print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))


##########################################
# 先使用pca再svm
#########################################
#pca先进行降维
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
# 利用训练特征决定(fit)20个正交维度的方向，并转化(transform)原训练特征
pca_X_train = pca.fit_transform(X_train)
# 利用上面得到的Model直接对test数据进行transform
pca_X_test = pca.transform(X_test)

svc = LinearSVC()
svc.fit(pca_X_train, y_train)
y_pca_svc_predict = svc.predict(pca_X_test)


##########################################
# 评估
#########################################

print(svc.score(pca_X_train, y_train))
print(classification_report(y_test, y_pca_svc_predict, target_names=np.arange(10).astype(str)))


















