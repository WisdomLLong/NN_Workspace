# -*- coding: utf-8 -*-

# DecisionTree，泰塔尼克号

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())

#------------------------------------------------------------------------------
# 数据预处理
#------------------------------------------------------------------------------
# 使用pandas，数据转入pandas独有的dataframe格式（二维数据表格），直接使用
# info()，查看数据的统计特性，仅仅是查看
titanic.info()

# 特征的选择，这个是十分重要的

X = titanic[['pclass', 'age', 'sex']]
y = titanic[['survived']]
X.info()

# 1）age的空白数据需要补全，用中位数或平均值
# 2）sex与pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替

X['age'].fillna(X['age'].mean(), inplace = True)

X.info()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                            random_state=33)

# 特征转换器，特征提取
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))

# 同样需要对测试数据进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

#------------------------------------------------------------------------------
# 使用决策树分类器
#------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
# 默认配置初始化决策树
dtc = DecisionTreeClassifier()
# 训练
dtc.fit(X_train, y_train)
# 预测
y_predict = dtc.predict(X_test)


#------------------------------------------------------------------------------
# 预测性能评估
#------------------------------------------------------------------------------
from sklearn.metrics import classification_report
print( dtc.score(X_test, y_test))
print( classification_report(y_predict, y_test, target_names=['died', 'survived']))








