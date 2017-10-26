#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

# execute parameters
index1 = sys.argv[1]  # raw data (train.csv)
index2 = sys.argv[2]  # test data (test.csv)
index3 = sys.argv[3]  # provided train feature (X_train)
index4 = sys.argv[4]  # provided test label (Y_train)
index5 = sys.argv[5]  # provided test feature (X_test)
index6 = sys.argv[6]  # prediction.csv

# gradient desecnet parameters
lr = 0.5
lamda = 0.005
max_iteration = 3000

# valid data percentage
percentage = 0.2


def csv_to_data(file, title=False):
    ''' data extraction'''
    if title:
        data = pd.read_csv(file, header=None)
    else:
        data = pd.read_csv(file)
    return data.as_matrix().astype('float')

########## Normalisation Methods ##########


def rescaling(data, feature):
    ''' three data normalisation methods'''
    feature_min = np.min(feature, axis=0)
    feature_max = np.max(feature, axis=0)
    data = (data - feature_min) / (feature_max - feature_min)
    return data


def mean_normalisation(data, feature):
    '''  three data normalisation methods'''
    feature_min = np.min(feature, axis=0)
    feature_max = np.max(feature, axis=0)
    feature_mean = np.mean(feature, axis=0)
    data = (data - feature_mean) / (feature_max - feature_min)
    return data


def standardization(data, feature):
    '''  three data normalisation methods'''
    feature_std = np.std(feature, axis=0)
    feature_mean = np.mean(feature, axis=0)
    data = (data - feature_mean) / feature_std
    return data
##########################################


def data_shuffle(feature, label):
    random = np.arange(len(feature))
    np.random.shuffle(random)
    return feature[random], label[random]


def split_valid_data(feature, label, percentage):
    feature_data_size = len(feature)
    valid_data_size = int(feature_data_size * percentage)
    feature, label = data_shuffle(feature, label)
    valid_feature, valid_label = feature[:
                                         valid_data_size], label[:valid_data_size]
    train_feature, train_label = feature[valid_data_size:-
                                         1], label[valid_data_size:-1]
    return train_feature, train_label, valid_feature, valid_label


def cross_entropy(label, predict):
    cross_entropy_loss = -(label * np.log(predict) +
                           (1 - label) * np.log(1 - predict))
    return np.sum(cross_entropy_loss)


def sigmoid(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1 - (1e-8))


def add_bias(data):
    return np.concatenate((data, np.ones(shape=(len(data), 1))), axis=1)


def gradient_descent(feature, label, lr, lamda, max_iteration):
    ''' gradient descent '''
    m, n = feature.shape  # m: vector number n: feature number
    w = np.zeros(shape=(n, 1))
    w_lr = np.zeros(shape=(n, 1))
    # iteration
    for i in range(max_iteration):
        predict = sigmoid(np.dot(feature, w))
        error = predict - label  # shape = (-1, 1)
        grad = np.dot(feature.T, error)

        # adaptive gradient descent
        w_lr = w_lr + grad ** 2

        # no regularization
        # w = w - lr / np.sqrt(w_lr) * grad
        # L1 regularization
        # w = w - lr / np.sqrt(w_lr) * (grad + lr * lamda * np.sign(w))
        # L2 regularization
        w = w * (1 - lr * lamda) - lr / np.sqrt(w_lr) * grad

        if (i + 1) % 100 == 0:
            predict_results = np.where(predict > 0.5, 1, 0)
            arrcuracy = sum(predict_results == label) / m
            loss = cross_entropy(label, predict) / m
            print(
                '=========Training Data Accuracy and Cross Entropy Loss at Epoch %d======' % (i + 1))
            print('[Train] accuracy = %f' % (arrcuracy))
            print('[Train] loss     = %f' % (loss))
    return w


def caculate_data(feature, w):
    result = sigmoid(np.dot(feature, w))
    label = np.where(result > 0.5, 1, 0)
    return label


def valid_data_accuracy(valid_calcu_label, valid_label):
    accuracy = sum(valid_label == valid_calcu_label) / len(valid_label)
    print('********Valid Data Accuracy********')
    print('[Valid] accuracy = %f' % accuracy)


def output_into_file(path, label):
    with open(path, 'w') as output:
        output.write('id,label\n')
        for i in range(len(label)):
            output.write(str(i + 1) + ',' + str(label[i][0]) + '\n')
    output.close()

########## Feature Model ##########


def model_0(feature):
    return feature


def model_1(feature):
    return np.concatenate((feature, feature**2), axis=1)


def model_2(feature):
    return np.concatenate((feature, feature**3), axis=1)


def model_3(feature):
    return np.concatenate((feature, feature**2, feature**3), axis=1)


def model_4(feature):
    select = [0, 1, 3, 4, 5]
    return np.concatenate((feature, feature[:, select]**2, feature[:, select]**3), axis=1)


def model_5(feature):
    select = [0, 1, 3, 4, 5]
    feature_select = feature[:, select]
    return np.concatenate((feature, feature_select ** 2, np.log(feature_select + 1e-2)), axis=1)


def model_6(feature):
    select = [0, 1, 3, 4, 5]
    feature_select = feature[:, select]
    return np.concatenate((feature,
                           feature_select ** 2,
                           feature_select ** 3,
                           feature_select ** 4,
                           feature_select ** 5,
                           np.log(feature_select + 1e-10),
                           (feature[:, 0] * feature[:, 3]).reshape(-1, 1),
                           (feature[:, 0] * feature[:, 5]).reshape(-1, 1),
                           ((feature[:, 0] * feature[:, 5]).reshape(-1, 1))**2,
                           (feature[:, 3] * feature[:, 5]).reshape(-1, 1),
                           (feature[:, 6] * feature[:, 5]).reshape(-1, 1),
                           (feature[:, 3] - feature[:, 4]).reshape(-1, 1),
                           ((feature[:, 3] - feature[:, 4]).reshape(-1, 1))**3
                           ), axis=1)

###################################


def main():
    ''' loading train data'''
    raw_train_feature = csv_to_data(index3)
    raw_train_label = csv_to_data(index4)

    ''' training data model transform '''
    model_train_feature = model_6(raw_train_feature)

    ''' training data normalisation '''
    normal_train_feature = standardization(
        model_train_feature, model_train_feature)

    ''' training data adding bias '''
    normal_train_feature_bias = add_bias(normal_train_feature)

    ''' split valid training data and execute training data'''
    shuffle_train_feature, shuffle_train_label, valid_feature, valid_label = split_valid_data(
        normal_train_feature_bias, raw_train_label, percentage)

    ''' execute training data '''
    w = gradient_descent(shuffle_train_feature,
                         shuffle_train_label, lr, lamda, max_iteration)
    print(w.shape)

    ''' valid training data '''
    valid_calcu_label = caculate_data(valid_feature, w)
    valid_data_accuracy(valid_calcu_label, valid_label)

    ''' loading test data '''
    raw_test_feature = csv_to_data(index5)

    ''' testing data model transform '''
    model_test_feature = model_6(raw_test_feature)

    ''' testing data normalisation and adding bias '''
    normal_test_feature = standardization(
        model_test_feature, model_train_feature)
    normal_test_feature_bias = add_bias(normal_test_feature)

    ''' caculate test data '''
    test_label = caculate_data(normal_test_feature_bias, w)
    output_into_file(index6, test_label)


if __name__ == '__main__':
    main()
