# by lnaBy 2017/4/11  kaggle.com.cn
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# op file
def load_train_data():
    data_list = []
    with open(r'Data/Digit_Recognizer_data/train.csv', 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            data_list.append(line)
    data_list.remove(data_list[0])
    data_list = np.array(data_list)
    train_x = data_list[:, 1:]
    train_y = data_list[:, 0]
    return normal(string_narr_to_int(train_x)), string_arr_to_int(train_y)


def load_test_data():
    data_list = []
    with open(r'Data/Digit_Recognizer_data/test.csv', 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            data_list.append(line)
    data_list.remove(data_list[0])
    test_x = np.array(data_list)
    return normal(string_narr_to_int(test_x))


# string to int
def string_narr_to_int(arr):
    row = arr.shape[0]
    col = arr.shape[1]
    new_arr = np.zeros(shape=[row, col])
    for i in range(row):
        for j in range(col):
            new_arr[i, j] = int(arr[i, j])
    return new_arr


# string to int
def string_arr_to_int(arr):
    row = arr.shape[0]
    new_arr = np.zeros(shape=[row])
    for i in range(row):
            new_arr[i] = int(arr[i])

    return new_arr


# normalization
def normal(arr):
    row = arr.shape[0]
    col = arr.shape[1]
    for i in range(row):
        for j in range(col):
            if arr[i, j] != 0:
                arr[i, j] = 1
    return arr


# save result csv
def save_result(test_y):
    m = test_y.shape[0]
    with open(r'Data/Digit_Recognizer_data/result.csv', 'w') as file:
        data_input = csv.writer(file)
        title = ["ImageId", "Label"]
        data_input.writerow(title)
        for i in range(m):
            temp = []
            temp.append(i)
            temp.append(test_y[i])
            data_input.writerow(temp)


# main
def main():
    train_x, train_y = load_train_data()
    test_x = load_test_data()
    print(train_y)
    print(train_y.shape)
    print(train_x)
    print(train_x.shape)
    print(test_x)
    print(test_x.shape)

    # train model
    knn_neigh = KNeighborsClassifier()
    knn_neigh.fit(train_x, train_y)

    # predict
    test_y = knn_neigh.predict(test_x)
    save_result(test_y)


# test function
if __name__ == "__main__":
    main()