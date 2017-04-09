# by lna 2017/3/8

print(__doc__)
import logging
from time import time
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
# logging.basicConfig(level=logging.INFO, format='%(osctime)s %(message)s')
# 数据处理
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_sample, h, w = lfw_people.images.shape
print("样本个数:", n_sample)
x = lfw_people.data
print("特征实例:", x)
n_c = x.shape[1]
n_r = x.shape[0]
print("矩阵大小:", x.shape)
print("特征个数:", n_c)
print("样本个数:", n_r)
y = lfw_people.target
print("标签个数:", len(y))
print("标签数据:", y)
target_name = lfw_people.target_names
print("名字:", target_name.shape[0])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 由于特征向量的维数比较多，用PCA降维 降到150个特征向量
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, x_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components,
          whiten=True).fit(x_train)

print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("降维之后的训练集：", x_train_pca.shape)
print("降维之后测试集：", x_test_pca.shape)
print("done in %0.3fs" % (time() - t0))

# 训练svm
