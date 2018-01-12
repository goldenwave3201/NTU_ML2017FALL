#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-30 19:46:29
# @Author  : Jintao Yu (yujintao2013@gmail.com)
# @Link    : https://jintaoyu.github.io

# io package
import sys
import numpy as np
import pandas as pd
# keras packages
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
# cluster fuction
from sklearn.cluster import KMeans

index1 = sys.argv[1]
index2 = sys.argv[2]
index3 = sys.argv[3]


dim = 32


def build_model():
    encoding_dim = dim
    input_img = Input(shape=(784,))
    # encoder layers
    encoded = Dense(784, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)

    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)
    # construct the encoder model for plotting
    encoder = Model(input=input_img, output=encoded)
    return autoencoder, encoder


def clustering(data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_


def get_result(test_data, x_label):
    result = []
    for i in range(len(test_data)):
        if x_label[test_data[i][0]] == x_label[test_data[i][1]]:
            result.append(1)
        else:
            result.append(0)
    return result


def result_to_csv(result, path):
    with open(path, 'w') as output:
        output.write("ID,Ans\n")
        for i in range(len(result)):
            output.write(str(i) + ',' + str(result[i]) + '\n')


def main():
    # training image
    images = np.load(index1)  # shape=(140000, 784)
    x_train = images.astype('float32') / 255. - 0.5

    # build model
    autoencoder, encoder = build_model()
    adam = Adam()
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=20,
                    batch_size=256, shuffle=True)

    # features by reducing dim
    encoded_imgs = encoder.predict(x_train)  # shape=(140000, 32)

    x_label, x_center = clustering(encoded_imgs)

    # testing case
    test_dataframe = pd.read_csv(index2)  # shape=(1980000, 3)
    test_data = test_dataframe.values[:, 1:]
    result = get_result(test_data, x_label)
    result_to_csv(result, index3)


if __name__ == '__main__':
    main()
