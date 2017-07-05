# -*- coding: utf-8 -*-

#@@ 线性多项式回归，并不是改变线性模型，而是使数据产生多维特征

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

from sklearn.linear_model import LinearRegression
# 导入numpy并且重命名为np
import numpy as np
# 在x轴上从0至25均匀采样100个数据点
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)


# 从sklearn.preprocessing导入多项式特征生成器
from sklearn.preprocessing import PolynomialFeatures
# 初始化4次多项式特征生成器
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)

# 从新映射绘图用x轴采样数据
xx_poly4 = poly4.transform(xx)
# 使用4次多项式回归模型对应x轴采样数据进行回归预测
yy_poly4 = regressor_poly4.predict(xx_poly4)



#--------------------------------------------------------------
# 画图
import matplotlib.pyplot as plt
# 分别对训练数据点、4次多项式回归曲线进行作图
plt.scatter(X_train, y_train)
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')
#@@ 为什么返回两个值？？？
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt4])
#@@ legend和handles都有什么作用？？？
plt.show()

print('4次多项式的R平方值是', regressor_poly4.score(X_train_poly4, y_train))


# 准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

X_test_poly4 = poly4.transform(X_test)
regressor_poly4.score(X_train_poly4, y_test)










