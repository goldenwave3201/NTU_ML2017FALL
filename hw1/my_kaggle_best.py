#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np

index1 = './train.csv'
index2 = sys.argv[1]
index3 = sys.argv[2]

# Read in training data
with open(index1, 'r', encoding='big5') as train_file:
    train_data = []
    for line in train_file:
        fields = line[:-1].split(',')
        train_data.append([0.0 if x in ['NR', ''] else float(x)
                           for x in fields[3:]])

# Prepare training data
train_data = np.array(train_data[1:])  # shape = (4320, 24)
train_data = train_data.reshape(12, -1, 24)  # shape = (12, 360, 24)
train_data = train_data.reshape(12, -1, 18, 24)  # shape = (12, 20, 18, 24)
train_data = train_data.swapaxes(1, 2).reshape(
    12, 18, -1)  # shape = (12, 18, 480)

# Remove Juli
train_data = np.concatenate((train_data[:6], train_data[7:]))
# print('train_data.shape =', train_data.shape)

needed_cols = [2, 7, 8, 9, 10, 12, 14, 15, 16, 17]
square_cols = [2, 7, 8, 9, 10, 12, 16, 17]
target_col = 9  # PM2.5
num_col = len(needed_cols)
prev_col = 9
X = []
Y = []
for mon in range(len(train_data)):
    for i in range(480 - prev_col):
        X.append(np.array([train_data[mon][col][i:i + prev_col]
                           for col in needed_cols]))
        Y.append(train_data[mon][target_col][i + prev_col])

sX = []
for mon in range(len(train_data)):
    for i in range(480 - prev_col):
        sX.append(np.array([train_data[mon][col][i:i + prev_col]
                            for col in square_cols]))

# Training
X = np.array(X)
sX = np.array(sX)
col_o3 = X[:, 1].reshape(len(X), prev_col)
col_pm2 = X[:, 3].reshape(len(X), prev_col)
col_mul = col_o3 * col_pm2
X = X.reshape(len(X), prev_col * num_col)
sX = sX.reshape(len(sX), prev_col * len(square_cols))
Y = np.array(Y)
X = np.concatenate((X, sX**2, col_mul), axis=1)

# Normalization
feat_max = np.max(X, axis=0)
feat_min = np.min(X, axis=0)
X = (X - feat_min) / (feat_max - feat_min + 1e-20)

train_X = X
train_Y = Y

# Declare basic settings
iteration = 40000
lr = 0.5
b_lr = 1e-20
w_lr = np.full((X.shape[1]), 1e-20)
b = 0.0
w = np.ones(shape=(X.shape[1]))
prev_rmse = 1e20
rising_cnt = 0
reg_lambda = 1e-3

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros(shape=(X.shape[1]))

    predictions = train_X.dot(w) + b
    errors = train_Y - predictions

    b_grad = b_grad - 2.0 * np.sum(errors)
    # w_grad = w_grad - 2.0 * np.dot(train_X.T, errors)
    w_grad = w_grad - 2.0 * np.dot(train_X.T, errors)

    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
    # Update parameters
    b = b - lr / np.sqrt(b_lr) * b_grad
    w = w - lr / np.sqrt(w_lr) * (w_grad - reg_lambda * w)

    if (i + 1) % 1000 == 0:
        print('Iterations: %d' % (i + 1))
        print('RMSE Loss = %f' % np.sqrt(np.mean(errors ** 2)))

np.set_printoptions(threshold=np.nan)


# Read in testing data
with open(index2, 'r', encoding='big5') as test_file:
    test_data = []
    for line in test_file:
        fields = line[:-1].split(',')
        test_data.append([0.0 if x in ['NR', ''] else float(x)
                          for x in fields[2:]])

# Prepare testing data
test_data = np.array(test_data)  # shape = (4320, 9)
test_data = test_data.reshape(-1, 18, 9)  # shape = (240, 18, 9)
test_X = []
test_Y = []
for i in range(test_data.shape[0]):
    test_X.append(np.array([test_data[i][col][9 - prev_col:]
                            for col in needed_cols]))
sX = []
for i in range(test_data.shape[0]):
    sX.append(np.array([test_data[i][col][9 - prev_col:]
                        for col in square_cols]))

test_X = np.array(test_X)
sX = np.array(sX)

# Testing
for i in range(len(test_X)):
    col_o3 = test_X[i][1].flatten()
    col_pm2 = test_X[i][3].flatten()
    col_mul = col_o3 * col_pm2
    x_flat = test_X[i].flatten()
    sx_flat = sX[i].flatten()

    x_flat = np.concatenate((x_flat, sx_flat ** 2, col_mul))
    x_flat = (x_flat - feat_min) / (feat_max - feat_min + 1e-20)
    test_Y.append(x_flat.dot(w) + b)

# Write predictions to output file
with open(index3, 'w') as output_file:
    outputs = ['id,value']
    for i in range(len(test_Y)):
        outputs.append('id_%d,%f' % (i, test_Y[i]))
    output_file.write('\n'.join(outputs) + '\n')
