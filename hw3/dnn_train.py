#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LeakyReLU, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


#--------input parameters--------
index1 = './data/train.csv'
index2 = './dnn_model'
# idnex1 = sys.argv[1]
# index2 = sys.argv[2]

#---------inital parameters------
optlr = 1e-4
epoch_num = 10


def str_to_float(string):
    # str type to float type
    str_data = string.split(' ')
    float_data = []
    for data in str_data:
        data = float(data)
        float_data.append(data)
    return float_data


def csv_to_X_Y(path):
    # load train.csv
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
    Y = np.array(Y)
    return X, Y


def split_data(X, Y):
    # split train set and valid data
    X_train, Y_train = X[:-4000], Y[:-4000]
    X_valid, Y_valid = X[-4000:], Y[-4000:]
    return X_train, Y_train, X_valid, Y_valid


def image_generator(X, Y):
    # keras.io exmaple
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=None,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    datagen.fit(X)
    img_gen = datagen.flow(X, Y, batch_size=32)
    return img_gen


def DNN(X_train, Y_train, X_valid, Y_valid, img_gen):
    # define a sequential model
    model = Sequential()

    # flatten the input
    model.add(Flatten(input_shape=(48, 48, 1)))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    model.summary()

    # configure the model for training
    opt = Adam(lr=optlr)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # train the model
    hist = model.fit_generator(
        img_gen,
        # keras.io example
        steps_per_epoch=len(X_train) / 32,
        epochs=epoch_num,
        validation_data=(X_valid, Y_valid)
    )

    model.save(index2)
    model.save("./model/dnn_model_lr=%s_epochs=%d" % (str(optlr), epoch_num))
    return hist


def plot_accuracy(hist, path):
    # plot accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.savefig(path)
    plt.close()


def plot_loss(hist, path):
    # plot loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('# of epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(path)
    plt.close()


def main():
    X_raw, Y_raw = csv_to_X_Y(index1)
    X_raw = X_raw / 255
    Y_raw = np_utils.to_categorical(Y_raw, 7)
    X_train, Y_train, X_valid, Y_valid = split_data(X_raw, Y_raw)
    img_gen = image_generator(X_train, Y_train)
    hist = DNN(X_train, Y_train, X_valid, Y_valid, img_gen)
    plot_accuracy(hist, './image/dnn_accuracy.png')
    plot_loss(hist, './image/dnn_loss.png')


if __name__ == '__main__':
    main()
