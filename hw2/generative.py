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


def csv_to_data(file, title=False):
    ''' data extraction'''
    if title:
        data = pd.read_csv(file, header=None)
    else:
        data = pd.read_csv(file)
    return data.as_matrix().astype('float')


def sigmoid(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1 - (1e-8))


def maximun_likehihood(X_train, Y_train, X_test):
    m, n = X_train.shape
    # calculate mu1 and mu2
    mu1 = np.zeros(shape=(1, n))
    mu2 = np.zeros(shape=(1, n))
    num1 = 0
    num2 = 0
    for i in range(m):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            num1 += 1
        else:
            mu2 += X_train[i]
            num2 += 1
    mu1 = mu1 / num1
    mu2 = mu2 / num2
    N1 = num1
    N2 = num2

    # calculate convariance matrix
    sigma1 = np.zeros(shape=(n, n))
    sigma2 = np.zeros(shape=(n, n))
    for j in range(m):
        if Y_train[j] == 1:
            sigma1 += np.dot((X_train[j] - mu1).T, X_train[j] - mu1)
        else:
            sigma2 += np.dot((X_train[j] - mu2).T, X_train[j] - mu2)
    sigma1 = sigma1 / N1
    sigma2 = sigma2 / N2
    shared_sigma = (float(N1 / m) * sigma1) + (float(N2 / m) * sigma2)

    # calculate w, b
    shared_sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot((mu1 - mu2), shared_sigma_inverse)
    b = -0.5 * np.dot(np.dot(mu1, shared_sigma_inverse), mu1.T) + 0.5 * \
        np.dot(np.dot(mu2, shared_sigma_inverse),
               mu2.T) + np.log(float(N1 / N2))
    return w, b


def caculate_data(X_test, w, b):
    result = sigmoid(np.dot(X_test, w.T) + b)
    label = np.where(result > 0.5, 1, 0)
    return label


def output_into_file(path, label):
    with open(path, 'w') as output:
        output.write('id,label\n')
        for i in range(len(label)):
            output.write(str(i + 1) + ',' + str(label[i][0]) + '\n')
    output.close()


def main():
    X_train = csv_to_data(index3)
    Y_train = csv_to_data(index4)
    X_test = csv_to_data(index5)
    w, b = maximun_likehihood(X_train, Y_train, X_test)
    label = caculate_data(X_test, w, b)
    output_into_file(index6, label)


if __name__ == '__main__':
    main()
