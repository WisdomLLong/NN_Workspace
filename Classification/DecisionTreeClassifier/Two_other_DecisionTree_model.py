# 这里相比于传统的决策树多了两类“随机森林”和“梯度提升决策树GTB”

import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())

#------------------------------------------------------------------------------
# 数据预处理
#------------------------------------------------------------------------------
X = titanic[['pclass','age','sex']]
y = titanic['survived']

X['age'].fillna(X['age'].mean(), inplace=True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))


#------------------------------------------------------------------------------
# 训练及预测
#------------------------------------------------------------------------------
# 使用常规的单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)


#------------------------------------------------------------------------------
# 预测性能对比
#------------------------------------------------------------------------------
from sklearn.metrics import classification_report

# 分别输出三种算法的精确率、召回率和F1指标
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

print('The accuracy of random forest classsifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('The accuracy of gradient ress boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))





