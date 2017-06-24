# -*- coding: utf-8 -*-

# 定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）
measurements = [{'city':'Dubai', 'temperature':33.},\
                {'city':'London', 'temperature':12.},\
                {'city':'ZhengZhou', 'temperature':26.}]
                
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

#@@ 
'''
仿真结果：
[[  1.   0.   0.  33.]
 [  0.   1.   0.  12.]
 [  0.   0.   1.  26.]]
['city=Dubai', 'city=London', 'city=ZhengZhou', 'temperature']

可以看出DictVectorizer对于类别型和数值型的特征处理方式是不同的
'''
