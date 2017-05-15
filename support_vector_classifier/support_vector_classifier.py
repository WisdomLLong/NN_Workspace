# -*- coding: utf-8 -*-
# 这里强调使用SVM的“分类”能力
# 对0~9的手写体数字进行分类

#--------------------------
# 手写体数据读取
#--------------------------
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)
# (1797,64)图像数据共1797条，每幅图片是8*8=64的像素矩阵

#--------------------------
# split数据
#--------------------------
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, \
        digits.target, test_size=0.25, random_state=33 )
# 检测数据集规模
print(y_train.shape)
print(y_test.shape)


#--------------------------
# 训练和识别
#--------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)   

lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)


#--------------------------
# 评估
#--------------------------
print('The Accuracy of Linear SVC is', lsvc.score(X_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=\
                            digits.target_names.astype(str)))


