#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import csv
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

index1 = sys.argv[1]
index2 = sys.argv[2]
rnn_model_path = './rnn.model'


# parameters
NB_WORDS = 20000
MAXLEN = 40
EMBEDDING_DIM = 100


def read_training_data_label(file):
    label = []
    texts = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' +++$+++ ')
            label.append(int(line[0]))
            texts.append(line[1].strip('\n'))
    return label, texts


def read_training_data_nolabel(file):
    texts = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line.strip('\n'))
    return texts


def read_testing_data(file):
    texts = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            start = line.find(',')
            texts.append(line[start + 1:].strip('\n'))
    return texts


def data_tokenize(texts, punc=False):
    if punc == False:
        tokenizer = Tokenizer(num_words=NB_WORDS,
                              filters='')
    else:
        tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index


def data_output(y_test, path):
    ans_file = open(path, 'w')
    ans_writer = csv.writer(ans_file)
    ans_writer.writerow(['id', 'label'])
    for i in range(200000):
        st = str(i)
        if y_test[i] < 0.5:
            y = 0
        else:
            y = 1
        ans_writer.writerow([st, y])
    ans_file.close()


def get_word_dict(model):
    word_dict = []
    for word, value in model.wv.vocab.items():
        word_dict.append(word)
    return word_dict


def transform_data(x, tokenizer):
    x_seq = tokenizer.texts_to_sequences(x)
    x_pad = pad_sequences(x_seq, maxlen=MAXLEN)
    return x_pad


def main():
    path1 = './txt/1.txt'
    path2 = './txt/2.txt'
    # training data with label
    label, training_data_label = read_training_data_label(path1)
    # training data without label
    training_data_nolabel = read_training_data_nolabel(path2)
    # testing data
    testing_data = read_testing_data(index1)

    doc = training_data_label + training_data_nolabel + testing_data

    w2v_model = Word2Vec.load('./w2v.model')
    w2v_word_dict = get_word_dict(w2v_model)

    tokenizer, word_index = data_tokenize(doc)
    x_test = transform_data(testing_data, tokenizer)
    rnn_model = load_model(rnn_model_path)
    y_test = rnn_model.predict(x_test)
    data_output(y_test, index2)


if __name__ == '__main__':
    main()
