# -*- coding: utf-8 -*-

# 对新闻文本进行分类处理

## from后面的一句话中，最后一个点前面的都是文件夹，后面的是一个py文件。
## import的是一个类，创建实体后，就可以运用其中的方法了
## import的或是一个方法，不用创建实体，可以直接用

from sklearn.datasets import fetch_20newsgroups
#与之前预测的数据不同，fetch_20newsgroups需要即时从互联网下载数据

#already = 1

#if already == 0:
#    print('*********************\n已经下载好新闻的数据了\n*********************')
#else:
news = fetch_20newsgroups(subset='all')   #unexpected indent 非理想缩进
    # 如何将news保存到本地？？？这样就可以根据路径和名字判断是否已经下载文件了
already = 0
    
print(len(news.data))
print(news.data[0])


#--------------------------
# 数据分隔
#--------------------------
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, \
                test_size= 0.25, random_state= 33)


#--------------------------
# 特征抽取、训练和预测
#--------------------------
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)


#--------------------------
# 评估
#--------------------------
from sklearn.metrics import classification_report
print('The accuracy of Navie Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))



