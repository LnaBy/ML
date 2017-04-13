# by lnaBy 2017/4/10

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# list
train_x = [[0, 0], [0, 0.2], [0, 0.5], [1, 1]]
train_y = [0, 0, 2, 1]

# array
train_x_to_arr = np.array(train_x)
print(train_x_to_arr)
train_y_to_arr = np.array(train_y)

# train model
k_neigh = KNeighborsClassifier(n_neighbors=2, algorithm='auto')
# k_neigh.fit(train_x, train_y)
k_neigh.fit(train_x_to_arr, train_y_to_arr)

test_x = [[1, 1]]
test_x_to_arr = np.array(test_x)
# predict
print("class:", k_neigh.predict(test_x_to_arr))
print("[class_1_proba,class_1_probe...]", k_neigh.predict_proba(test_x_to_arr))
print("[point,distance]:", k_neigh.kneighbors(test_x_to_arr))

