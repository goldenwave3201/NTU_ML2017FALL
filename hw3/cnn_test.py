#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from keras.models import load_model

#--------input parameters----------
# model path
index1 = './cnn_model'
# testing data path
index2 = sys.argv[1]
# prediction file path
index3 = sys.argv[2]


def csv_to_X_Y(path):
  data_frame = pd.read_csv(path)
  data = np.array(data_frame)
  m = data.shape[0]
  X = []
  Y = []
  for i in range(m):
    Y.append(data[i][0])
    X.append(str_to_float(data[i][1]))
  X = np.array(X)
  X = np.reshape(X, [-1, 48, 48, 1])
  X = X / 255
  Y = np.array(Y)
  return X, Y


def str_to_float(string):
  str_data = string.split(' ')
  float_data = []
  for data in str_data:
    data = float(data)
    float_data.append(data)
  return float_data


def calculation(model, X):
  return model.predict(X)


def output_into_file(path, Y):
  with open(path, 'w') as output:
    output.write('id,label\n')
    for i in range(len(Y)):
      output.write(str(i) + ',' + str(Y[i]) + '\n')
  output.close()


def main():
  X_test, X_test_id = csv_to_X_Y(index2)
  model = load_model(index1)
  Y_test = calculation(model, X_test)
  result = np.argmax(Y_test, axis=-1)
  output_into_file(index3, result)


if __name__ == '__main__':
  main()
