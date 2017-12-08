#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import string
from gensim.models.word2vec import Word2Vec

index1 = './txt/1.txt'
index2 = './txt/2.txt'
index3 = './txt/3.txt'


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


def split_sentences(texts):
    sentences = [s.split() for s in texts]
    return sentences


def main():
    # training data with label
    label, training_data_label = read_training_data_label(index1)

    # training data without label
    training_data_nolabel = read_training_data_nolabel(index2)

    # testing data
    testing_data = read_testing_data(index3)

    data = training_data_label + training_data_nolabel + testing_data
    sentences = split_sentences(data)
    w2v_model = Word2Vec(sentences, min_count=1)
    w2v_model.save('./w2v.model')


if __name__ == '__main__':
    main()
