# -*- coding: utf-8 -*-

# K-means算法对生物物种鸢yuan尾进行分类

from sklearn.datasets import load_iris
iris = load_iris()

#@@ 查看数据说明，是一个好习惯
print(iris.DESCR)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,\
                test_size = 0.25, random_state = 33)


#--------------------------
# 特征标准化、训练和预测
#--------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)  #@@返回的仍是实例knc本身，只是其中的参数变化了
y_predict = knc.predict(X_test)


#--------------------------
# 评估
#--------------------------
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = iris.target_names))




