#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import sys


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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


def main():
    X_raw, Y_raw = csv_to_X_Y('../data/train.csv')
    X_raw = X_raw / 255
    X_train, Y_train, X_valid, Y_valid = split_data(X_raw, Y_raw)

    model_path = '../cnn_model'
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats = X_valid.reshape(-1, 48, 48, 1)
    predictions = emotion_classifier.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    print (predictions)
    te_labels = Y_valid
    print (te_labels)
    conf_mat = confusion_matrix(te_labels, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=[
                          "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    plt.savefig('./problem3.png')
    plt.show()


if __name__ == '__main__':
    main()
