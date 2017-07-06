# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
import numpy as np

news = fetch_20newsgroups(subset='all')
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], \
        news.target[:3000], test_size=0.25, random_state=33)

from sklearn.svm import SVC
# 导入文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入 Pipeline
from sklearn.pipeline import Pipeline

# 使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english',analyzer='word')), ('svc', SVC())])


# 这里需要验证的2个超参数的个数分别是4/3， svc_gamma的参数共有10^-2， 10^-1...。
# 这样我们一共有12种的超参数组合，12个不同参数下的模型
parameters = {'svc__gamma':np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

# 网络搜索模块 GridSearchCV
from sklearn.grid_search import GridSearchCV

# 将12组参数组合以及初始化的Pipeline包括3折交叉验证的要求全部告知GridSearchCV
# 注意refit=True这样一个设定
# 此处为单线程运行
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 此处为并行，n_jobs=-1代表使用该计算机全部CPU
gs_parallel = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)


# 执行单线程网络搜索 
#@@ 注意这里%time只能在IPython 的console 界面使用
#%time _=gs.fit(X_train, y_train)
#gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
#print(gs.score(X_test, y_test))













