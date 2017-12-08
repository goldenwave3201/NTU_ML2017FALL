#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

index1 = sys.argv[1]
index2 = sys.argv[2]
index3 = './txt/3.txt'
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
    #sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index


def get_embedding_matrix(model, word_index):
    embedding_matrix = np.zeros((NB_WORDS + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        st = ''
        for j in range(3 - len(word)):
            st += ' '
        if i > NB_WORDS:
            continue
        embedding_matrix[i] = model.wv[word]
    return embedding_matrix


def train(x_train, y_train, embedding_matrix):
    model = Sequential()
    model.add(Embedding(NB_WORDS + 1,
                        EMBEDDING_DIM,
                        input_length=MAXLEN,
                        weights=[embedding_matrix],
                        trainable=False))
    model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    h = model.fit(x_train, y_train,
                  epochs=20,
                  batch_size=1024,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=True)
    model.save(rnn_model_path)
    return h


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

    # training data with label
    label, training_data_label = read_training_data_label(index1)
    # training data without label
    training_data_nolabel = read_training_data_nolabel(index2)
    # testing data
    testing_data = read_testing_data(index3)

    w2v_model = Word2Vec.load('./w2v.model')
    w2v_word_dict = get_word_dict(w2v_model)

    doc = training_data_label + training_data_nolabel + testing_data
    tokenizer, word_index = data_tokenize(doc)
    embedding_matrix = get_embedding_matrix(w2v_model, word_index)

    x_train_label = transform_data(training_data_label, tokenizer)
    y_train_label = np.asarray(label)
    h = train(x_train_label, y_train_label, embedding_matrix)


if __name__ == '__main__':
    main()
