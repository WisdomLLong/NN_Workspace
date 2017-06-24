# -*- coding: utf-8 -*-
# 两种特征抽的方法，这是相对于CountVectorizer方法的第二类, 同样是新闻处理

from sklearn.datasets import fetch_20newsgroups
# 从网上及时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')
from sklearn.cross_validation import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(news.data, news.target,\
                            test_size = 0.25, random_state = 33)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
"""
如果这里要使用停用词
count_vec = TfidfVectorizer(analyzer='word', stop_words='english')
"""
X_tfidf_train = tfidf.fit_transform(X_train)
X_tfidf_test = tfidf.transform(X_test)



from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
# NB进行训练
mnb_count.fit(X_tfidf_train, y_train)

#输出模型的准确性
print('The accuracy of the NB with TfidfVectorizer:', \
      mnb_count.score(X_tfidf_test, y_test))

# 用classification_report进行详细评估
y_predict = mnb_count.predict(X_tfidf_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=news.target_names))













