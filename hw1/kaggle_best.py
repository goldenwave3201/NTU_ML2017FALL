#!/usr/bin/env python3

import sys
import numpy as np
import os

index = './data/train.csv'
# testing data path
index1 = sys.argv[1]
# output data path
index2 = sys.argv[2]

# feature parameter setting
select_hour = 9
select_column = 9
need_column = [2, 7, 8, 9, 10, 12, 14, 15, 16, 17]
need_column2 = [2, 7, 8, 9, 10, 12, 16, 17]
need_num = len(need_column)

# gradient descent parameters setting
lamda = 1e-3      # regularization rate
lr = 0.5
max_iteration = 40000


def traincsv_to_traindata(path):
    with open(path, 'r', encoding='big5') as train_file:
        train_data = []
        for line in train_file:
            fields = line[:-1].split(',')
            train_data.append([0.0 if x in ['NR', ''] else float(x)
                               for x in fields[3:]])
            # print(fields)
        # shape an array: (18*20*12, 24) = (4320, 24)
        train_data = np.array(train_data[1:])
        train_data = train_data.reshape(12, 20, 18, 24)
        train_data = train_data.swapaxes(1, 2).reshape(12, 18, 480)
        # print(train_data) shape = (12, 18, 480)
        train_data = np.concatenate((train_data[:6], train_data[7:]))
    return train_data


def get_specific_features_in_9_hours(train_data):
    train_feature1 = []
    train_feature2 = []
    train_feature = []
    train_label = []
    # mon: month number
    for mon in range(len(train_data)):
        # m: feature sets number every month (480 - 9 = 471)
        for num in range(480 - select_hour):
            train_feature1.append(
                np.array([train_data[mon][col][num: num + select_hour] for col in need_column]))
            train_label.append(
                train_data[mon][select_column][num + select_hour])
    for mon in range(len(train_data)):
        # m: feature sets number every month (480 - 9 = 471)
        for num in range(480 - select_hour):
            train_feature2.append(
                np.array([train_data[mon][col][num: num + select_hour] for col in need_column2]))

    train_feature1 = np.array(train_feature1)  # shape = (5181, 10, 9)
    train_feature2 = np.array(train_feature2)  # shape = (5181, 8, 9)
    #print(train_feature1.shape, train_feature2.shape)
    train_label = np.array(train_label)

    o3_col = train_feature1[:, 1].reshape(len(train_feature1), select_hour)
    pm25_col = train_feature1[:, 3].reshape(len(train_feature1), select_hour)
    mul_col = o3_col * pm25_col  # shape = (5181, 9)
    # print(mul_col.shape)
    train_feature1 = train_feature1.reshape(len(train_feature1), -1)
    train_feature2 = train_feature2.reshape(len(train_feature2), -1)

    train_feature = np.concatenate(
        (train_feature1, train_feature2**2, mul_col), axis=1)
    # train_feature shape (5181, 171) train_label shape (5181,)
    return train_feature, train_label


def normalize_data(data, feature):
    feature_max = np.max(feature, axis=0)
    feature_min = np.min(feature, axis=0)
    data = (data - feature_min) / (feature_max - feature_min + 1e-20)
    return data


def gradient_descent(train_feature, train_label, lr, lamda, max_iteration):
    N = train_feature.shape[1]
    # initial w, b
    w = np.ones(shape=N)
    b = np.zeros(shape=1)
    w_lr = np.full(N, 1e-20)
    b_lr = np.full(1, 1e-20)

    # print(w, '\n', w.shape, '\n', b, '\n', b.shape)

    for i in range(max_iteration):
        w_grad = np.zeros(shape=N)
        b_grad = 0.0

        predict = np.dot(train_feature, w) + b
        # print(predict.shape)
        error = train_label - predict
        # print(error, error.shape)

        # calculate w_grad, b_grad
        w_grad = w_grad - 2.0 * np.dot(error, train_feature)
        # print(w_grad.shape)
        b_grad = b_grad - 2.0 * np.sum(error)

        # update w_lr, b_lr
        w_lr = w_lr + w_grad ** 2
        b_lr = b_lr + b_grad ** 2

        # update w, b
        w = w - lr / np.sqrt(w_lr) * (w_grad - lamda * w)
        b = b - lr / np.sqrt(b_lr) * b_grad
        # print(w, '\n', b, '\n')

        if (i + 1) % 1000 == 0:
            print('iterations = %d' % (i + 1))
            print('RMSE Loss = %f' %
                  np.sqrt(np.mean(error ** 2)))

    return w, b


def w_b_to_model_csv(w, b):
    with open('./kaggle_best_model.csv', 'w') as model_csv:
        for i in range(len(w)):
            model_csv.write('w_%d=' % (i + 1))
            model_csv.write(str(w[i]) + '\n')
        for j in range(len(b)):
            model_csv.write('b=')
            model_csv.write(str(b[j]) + '\n')
    model_csv.close()


def testcsv_to_testdata(path):
    with open(path, 'r', encoding='big5') as test_file:
        test_data = []
        for line in test_file:
            fields = line[:-1].split(',')
            test_data.append([0.0 if x in ['NR', ''] else float(x)
                              for x in fields[2:]])
        test_data = np.array(test_data)
        test_data = test_data.reshape(-1, 18, 9)
    return test_data


def testdata_to_feature(test_data):
    test_feature1 = []
    test_feature2 = []
    test_feature = []
    for i in range(test_data.shape[0]):
        test_feature1.append(
            np.array([test_data[i][col][9 - select_hour:] for col in need_column]))
    for i in range(test_data.shape[0]):
        test_feature2.append(
            np.array([test_data[i][col][9 - select_hour:] for col in need_column2]))
    test_feature1 = np.array(test_feature1)  # shape = (240, 10, 9)
    test_feature2 = np.array(test_feature2)  # shape = (240, 8, 9)
    #print(test_feature1.shape, test_feature2.shape)
    o3_col = test_feature1[:, 1].reshape(len(test_feature1), -1)
    pm25_col = test_feature1[:, 3].reshape(len(test_feature1), -1)
    mul_col = o3_col * pm25_col

    test_feature1 = test_feature1.reshape(len(test_feature1), -1)
    test_feature2 = test_feature2.reshape(len(test_feature2), -1)

    test_feature = np.concatenate(
        (test_feature1, test_feature2**2, mul_col), axis=1)
    return test_feature


def calculate_test_data(w, b, test_feature):
    test_label = np.dot(test_feature, w) + b
    return test_label


def output_write_into_file(path, test_label):
    with open(path, 'w') as output_file:
        output_file.write('id,value\n')
        for i in range(len(test_label)):
            output_file.write('id_' + str(i) + ',' + str(test_label[i]) + '\n')
    output_file.close()


def model_csv_to_w_b(model_csv):
    with open(model_csv, 'r') as model_csv:
        w = []
        b = []
        wb = []
        for line in model_csv:
            fields = line[:-1].split('=')
            wb.append(float(fields[1]))
    wb = np.array(wb)
    w = wb[:-1]
    b = np.array([wb[-1]])
    return w, b


def main():
    train_data = traincsv_to_traindata(index)  # shape = (11, 18, 480)
    train_feature, train_label = get_specific_features_in_9_hours(train_data)
    # print(train_feature.shape)
    normal_train_feature = normalize_data(train_feature, train_feature)
    w, b = gradient_descent(normal_train_feature,
                            train_label, lr, lamda, max_iteration)
    w_b_to_model_csv(w, b)
    test_data = testcsv_to_testdata(index1)
    test_feature = testdata_to_feature(test_data)
    # print(test_feature)
    normal_test_feature = normalize_data(test_feature, train_feature)
    #print(normal_test_feature, normal_test_feature.shape)
    test_label = calculate_test_data(w, b, normal_test_feature)
    output_write_into_file(index2, test_label)


def main2():
    train_data = traincsv_to_traindata(index)  # shape = (11, 18, 480)
    train_feature, train_label = get_specific_features_in_9_hours(train_data)
    # print(train_feature.shape)
    w, b = model_csv_to_w_b('./kaggle_best_model.csv')
    test_data = testcsv_to_testdata(index1)
    test_feature = testdata_to_feature(test_data)
    # print(test_feature)
    normal_test_feature = normalize_data(test_feature, train_feature)
    #print(normal_test_feature, normal_test_feature.shape)
    test_label = calculate_test_data(w, b, normal_test_feature)
    output_write_into_file(index2, test_label)


if __name__ == '__main__':
    is_file = os.path.isfile('./kaggle_best_model.csv')
    # print(is_file)
    if (False == is_file):
        main()
    else:
        main2()
