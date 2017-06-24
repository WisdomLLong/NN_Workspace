# -*- coding: utf-8 -*-
# 特征抽取之后使用朴素贝叶斯Navie Bayes分类器进行分类
# 导入20类新闻文本数据抓取器
#@@ 停用词：为提高文本搜索效率，在处理自然语言数据或文本之前需要滤掉的词，例如：the, a, there
from sklearn.datasets import fetch_20newsgroups
# 从网上及时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(news.data, news.target,\
                            test_size = 0.25, random_state = 33)

from sklearn.feature_extraction.text import CountVectorizer
# 默认配置不去出英文停用词
count_vec = CountVectorizer()

"""
如果这里要使用停用词
count_vec = CountVectorizer(analyzer='word', stop_words='english')
"""

X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
#@@ X_count_test类型是scipy.sparse.csr.csr_matrix，csr指compressed sparse row matrix
#@@ 存储方式 (新闻标号，单词标号) 出现次数


from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
# NB进行训练
mnb_count.fit(X_count_train, y_train)

#输出模型的准确性
print('The accuracy of the NB with CountVecorizer:', \
      mnb_count.score(X_count_test, y_test))

# 用classification_report进行详细评估
y_predict = mnb_count.predict(X_count_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=news.target_names))












