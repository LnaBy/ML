# by lna 2017/3/7 具体的一个svm的实现并画出图形

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
colors  = "rb"
# 产生一些数据二维矩阵
np.random.seed(0)
# 产生一些正态分布的数据并且加减2
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]

# 设置lables
y = [0] * 20 + [1] * 20
Y = np.array(y)

# 训练svm 线性核函数
clf = svm.SVC(kernel="linear")
clf.fit(x, y)
# 测试数据
z = np.r_[np.random.randn(3, 2) - [2, 2], np.random.randn(3, 2) + [2, 2]]
print(clf.predict(z))
# 下面是得到超平面
w = clf.coef_  # 得到权重
a = -w[0][0] / w[0][1]
xx = np.linspace(-5, 5)
# clf.intercept是常数
yy = a * xx - (clf.intercept_[0]) / w[0][1]

# 得到上下与超平面平行的线
# 下面2个表达式都可以,其实是一样的
# b = clf.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])
yy_down = a * (xx - clf.support_vectors_[0][0]) + clf.support_vectors_[0][1]
# b = clf.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])
yy_up = a * (xx - clf.support_vectors_[-1][0]) + clf.support_vectors_[-1][1]

plt.plot(xx, yy, 'g')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
# 画图
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='y')
for i, color in zip(range(2), colors):
    idx = np.where(Y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color)
plt.show()