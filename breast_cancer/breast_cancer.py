# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


#--------------------------
# 数据预处理
#--------------------------

#创建特征列表
column_names=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell',
              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
              'Class']
#使用pandas.read.csv函数从互联网读取指定数据
#代码太长，在要换行的地方加上、
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names= column_names) 
#将？替代为标准缺失值表示
data= data.replace(to_replace='?', value= np.nan)                           
#丢弃带有缺省值的数据（只要有一个维度有缺失）
data= data.dropna(how= 'any')
#输出data的数据量和纬度
print(data.shape)


#--------------------------
# 准备训练、测试数据
#--------------------------
#使用sklearn.cross_valiation里的train_test_split模块用于分隔数据
from sklearn.cross_validation import train_test_split
#随机采样25%的数据用于测试，剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test= train_test_split(data[column_names[1:10]], \
    data[column_names[10]], test_size=0.25, random_state=33)
#查验训练样本的数量和类别分布
print(y_train.value_counts())
#查验测试样本的数量和类别分布
print(y_test.value_counts())


#--------------------------
# 训练和预测
#--------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.linear_model import SGDClassifier # 随机梯度参数估计
# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果
# 不会被某些维度过大的特征值而主导
ss= StandardScaler()
X_train= ss.fit_transform(X_train)
X_test= ss.transform(X_test)

lr = LogisticRegression()
sgdc= SGDClassifier()

# 调用LogisticRegression中的fit函数/模型用来训练模型参数
lr.fit(X_train, y_train)
# 模型lr对X_test进行预测
lr_y_predict= lr.predict(X_test)

sgdc.fit(X_train, y_train)
sgdc_y_predict= sgdc.predict(X_test)


#--------------------------
# 性能分析
#--------------------------
from sklearn.metrics import classification_report

#使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
#利用classification_report模块获得三个指标的结果
print( classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']) )




