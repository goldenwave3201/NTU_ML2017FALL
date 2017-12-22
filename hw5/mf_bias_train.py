#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# matrix fatorization with the bias

import sys
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add
from keras.optimizers import Adam


index1 = './data/train.csv'
index2 = './data/movies.csv'
index3 = './data/users.csv'
index4 = './model/mf_bias.model'

# parameters
lat_dim_num = 8  # latent dimension


def get_max_id(data):
    # max user and movie id
    max_user_id = data['UserID'].drop_duplicates().max()
    max_movie_id = data['MovieID'].drop_duplicates().max()
    return max_user_id, max_movie_id


def get_tarin_feature(data):
    user = data['UserID'].values - 1
    movie = data['MovieID'].values - 1
    rating = data['Rating'].values
    return user, movie, rating


def split_data(user, movie, rating, ratio=0.1):
    # shuffle list
    indices = np.arange(user.shape[0])
    np.random.shuffle(indices)
    # shuffle data
    shuffled_user = user[indices]
    shuffled_movie = movie[indices]
    shuffled_rating = rating[indices]
    # valid num
    valid_num = int(ratio * shuffled_user.shape[0])
    # spilt data
    user_train = shuffled_user[valid_num:]
    user_valid = shuffled_user[:valid_num]
    movie_train = shuffled_movie[valid_num:]
    movie_valid = shuffled_movie[:valid_num]
    rating_train = shuffled_rating[valid_num:]
    rating_valid = shuffled_rating[:valid_num]
    return user_train, user_valid, movie_train, movie_valid, rating_train, rating_valid


def normalize(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    feature_normalized = (feature - mean) / std
    return feature_normalized, mean, std


def build_model(user_num, movie_num, lat_dim):
    # user
    user_input = Input(shape=(1,))
    user_embed = Embedding(user_num, lat_dim,
                           embeddings_initializer='random_normal')(user_input)
    user_embed = Flatten()(user_embed)
    user_bias = Embedding(
        user_num, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    # movie
    movie_input = Input(shape=(1, ))
    movie_embed = Embedding(movie_num, lat_dim,
                            embeddings_initializer='random_normal')(movie_input)
    movie_embed = Flatten()(movie_embed)
    movie_bias = Embedding(
        movie_num, 1, embeddings_initializer='zeros')(movie_input)
    movie_bias = Flatten()(movie_bias)
    # output
    dot = Dot(axes=1)([user_embed, movie_embed])
    output = Add()([dot, user_bias, movie_bias])
    model = Model(inputs=[user_input, movie_input], outputs=output)
    return model


def main():
    # init
    training_data = pd.read_csv(index1, sep=',')
    user, movie, rating = get_tarin_feature(training_data)
    max_user_id, max_movie_id = get_max_id(training_data)
    ratings_normalized, rating_mean, rating_std = normalize(rating)
    #print("rating_mean:%f" % (rating_mean))
    #print("rating_std:%f" % (rating_std))

    # split
    user_train, user_valid, movie_train, movie_valid, rating_train, rating_valid = split_data(
        user, movie, ratings_normalized)
    # model
    model = build_model(max_user_id, max_movie_id, lat_dim_num)
    opt = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    model.fit([user_train, movie_train], rating_train, epochs=5,
              validation_data=([user_valid, movie_valid], rating_valid), batch_size=128)
    model.save(index4)

    test_path = './data/test.csv'
    test = pd.read_csv(test_path)
    output = model.predict(
        [test['UserID'].values - 1, test['MovieID'].values - 1])

    output = output * rating_std + rating_mean
    with open('./result/mf_bias_result.csv', 'w') as f:
        f.write("TestDataID,Rating\n")
        for i, rating in enumerate(output):
            r = rating[0]
            if r > 5:
                r = 5
            if r < 1:
                r = 1
            f.write("{},{}\n".format(i + 1, r))


if __name__ == '__main__':
    main()
