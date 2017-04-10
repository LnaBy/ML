# by lna 2017/3 /7

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel="linear")
print(clf)
clf.fit(x, y)
print(clf.predict([[0.5, 0.6]]))
# 打印支持向量
print("支持向量：", clf.support_vectors_)
print("支持向量下标:", clf.support_)
print("每类支持向量的个数:", clf.n_support_)
X = np.array(x)
plt.axis([0, 4, 0, 4])

plt.scatter(X[:, 0], X[:, 1])
plt.show()