# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

###############################################################################
#  案例一：使用TensorFlow输出一句话
###############################################################################

# 初始化一个TensorFlow的常量：Hello Google Tensorflow!字符串，并命名为greeting
# 作为一个计算模块
greeting = tf.constant('Hello Google Tensorflow! ')

# 启动一个会话
sess = tf.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出会话执行的结果
print(result)
# 关闭会话。只是一种显示关闭会话的方式
sess.close()


###############################################################################
#  案例一：完成一次线性函数的计算
###############################################################################
# 声明matrixl为TensorFlow的一个1*2的行向量
matrixl = tf.constant([[3., 3.]])

# 声明matrixl为TensorFlow的一个2*1的列向量
matrix2 = tf.constant([[2.], [2.]])

# product将上述两个算子相乘，作为新算例
product = tf.matmul(matrixl, matrix2)

#继续将product与一个标量2.0求和拼接，作为最终的linear算例
linear = tf.add(product, tf.constant(2.0))

# 直接在会话中执行linear算例，相当于将上面所有的单独算例拼接成流程图来执行
with tf.Session() as sess:
    result = sess.run(linear)
    print(result)





