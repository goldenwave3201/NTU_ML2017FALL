#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from sklearn import tree, ensemble, metrics

# execute parameters
index1 = sys.argv[1]  # raw data (train.csv)
index2 = sys.argv[2]  # test data (test.csv)
index3 = sys.argv[3]  # provided train feature (X_train)
index4 = sys.argv[4]  # provided test label (Y_train)
index5 = sys.argv[5]  # provided test feature (X_test)
index6 = sys.argv[6]  # prediction.csv

# valid set percentage
percentage = 0.5


def csv_to_data(file, integer=False):
    ''' data extraction'''
    if integer:
        data = pd.read_csv(file).as_matrix().astype('int')
    else:
        data = pd.read_csv(file).as_matrix().astype('float')
    return data


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


def gradient_boosting_classifier(X_train, Y_train, X_test):
    gbc = ensemble.GradientBoostingClassifier()
    gbc = gbc.fit(X_train, Y_train.ravel())
    Y_test = gbc.predict(X_test)
    return Y_test


def output_into_file(path, label):
    with open(path, 'w') as output:
        output.write('id,label\n')
        for i in range(len(label)):
            output.write(str(i + 1) + ',' + str(label[i]) + '\n')
    output.close()


def main():
    X_train = csv_to_data(index3)
    Y_train = csv_to_data(index4, integer=True)

    X_train_shuffle, Y_train_shuffle = data_shuffle(X_train, Y_train)
    X_train_select, Y_train_select, X_train_valid, Y_train_valid = split_valid_data(
        X_train_shuffle, Y_train_shuffle, percentage)
    #print(X_train_select.shape, Y_train_select.shape)
    #print(X_train_valid.shape, Y_train_valid.shape)

    Y_train_valid_predict = gradient_boosting_classifier(
        X_train, Y_train, X_train_valid)
    print("Accuracy : %.4g" % metrics.accuracy_score(
        Y_train_valid, Y_train_valid_predict))
    # test data loading
    X_test = csv_to_data(index5)
    Y_test = gradient_boosting_classifier(
        X_train, Y_train, X_test)
    output_into_file(index6, Y_test)


if __name__ == '__main__':
    main()
