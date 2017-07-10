# -*- coding: utf-8 -*-
# 用20类新闻文本进行词向量训练
# 从sklearn.datasets导入20类新闻文本抓取器

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


from bs4 import BeautifulSoup
#@@ Beautiful Soup 是用Python写的一个HTML/XML的解析器，
#@@ 它可以很好的处理不规范标记并生成剖析树(parse tree)。 
#@@ 它提供简单又常用的导航（navigating），搜索以及修改剖析树的操作。
#@@ 它可以大大节省你的编程时间。
import nltk, re
# 定义一个函数名为news_to_sentences将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]','',sent.lower().strip()).split())
    return sentences
    
sentences = []

# 将长篇新闻文本中的句子剥离出来，用于训练
for x in X:
    sentences += news_to_sentences(x)

from gensim.models import word2vec

# 配置词向量的维度
num_features = 300
# 保证被考虑的词汇的频度
min_word_count = 20
# 设定并行化训练使用CPU计算核心的数量，多核可用
num_workers = 2
# 定义训练词汇量的上下文窗口大小
context = 5
downsampling = 1e-3

# 训练词向量模型
model = word2vec.Word2Vec(sentences, workers=num_workers,\
        size=num_features, min_count = min_word_count,\
        window = context, sample=downsampling)

# 这个设定代表当前训练好的词向量为最终版，也可以加快模型的训练速度
model.init_sims(replace=True)

# 利用训练好的模型，寻找训练文本中与morning最相关的10个词汇
model.most_similar('morning')
#@@ 可以通过print(model.wv.vocab)查看model的vocab，上面的morning是不在里面的，因此出错







