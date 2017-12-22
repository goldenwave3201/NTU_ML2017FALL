#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# dnn with the bias

import os
import sys
import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model

split_ratio = 0.1
DROPOUT_RATE = 0.5
lat_dim = 128

index1 = './data/train.csv'
index2 = './data/movies.csv'
index3 = './data/users.csv'
index4 = './data/test.csv'

model_path = './model/dnn_bias.model'


def build_model(num_users, num_movies, lat_dims, users_info, movies_info):
    """ Build keras model for training """
    # Model structure
    u_emb_input = Input(shape=(1, ))
    u_emb = Embedding(num_users, lat_dims,
                      embeddings_initializer='random_normal',
                      trainable=True)(u_emb_input)
    u_emb = Flatten()(u_emb)
    u_bias = Embedding(num_users, 1,
                       embeddings_initializer='zeros',
                       trainable=True)(u_emb_input)
    u_bias = Flatten()(u_bias)
    u_info_emb = Embedding(num_users,
                           users_info.shape[1],
                           weights=[users_info],
                           trainable=False)(u_emb_input)
    u_info_emb = Flatten()(u_info_emb)
    m_emb_input = Input(shape=(1, ))
    m_emb = Embedding(num_movies, lat_dims,
                      embeddings_initializer='random_normal',
                      trainable=True)(m_emb_input)
    m_emb = Flatten()(m_emb)
    m_bias = Embedding(num_movies, 1,
                       embeddings_initializer='zeros',
                       trainable=True)(m_emb_input)
    m_bias = Flatten()(m_bias)
    m_info_emb = Embedding(num_movies,
                           movies_info.shape[1],
                           weights=[movies_info],
                           trainable=False)(m_emb_input)
    m_info_emb = Flatten()(m_info_emb)
    u_emb = Dropout(DROPOUT_RATE)(u_emb)
    m_emb = Dropout(DROPOUT_RATE)(m_emb)
    concat = Concatenate()([u_emb, m_emb, u_info_emb, m_info_emb])
    dnn = Dense(256, activation='relu')(concat)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dense(256, activation='relu')(dnn)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dense(256, activation='relu')(dnn)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    output = Dense(1, activation='relu')(dnn)
    model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)

    return model


def main():
    ratings_df = pd.read_csv(index1)
    max_user_id = ratings_df['UserID'].drop_duplicates().max()
    max_movie_id = ratings_df['MovieID'].drop_duplicates().max()

    users = ratings_df['UserID'].values - 1
    movies = ratings_df['MovieID'].values - 1
    ratings = ratings_df['Rating'].values

    users_df = pd.read_csv(index3, sep='::', engine='python')
    users_age = (users_df['Age'] - np.mean(users_df['Age'])
                 ) / np.std(users_df['Age'])
    movies_df = pd.read_csv(index2, sep='::', engine='python')

    all_genres = np.array([])
    for genres in movies_df['Genres']:
        for genre in genres.split('|'):
            all_genres = np.append(all_genres, genre)
    all_genres = np.unique(all_genres)

    users_info = np.zeros((max_user_id, 23))
    movies_info = np.zeros((max_movie_id, all_genres.shape[0]))

    for idx, user_id in enumerate(users_df['UserID']):
        gender = 1 if users_df['Gender'][idx] == 'M' else 0
        occu = np.zeros(np.max(np.unique(users_df['Occupation'])) + 1)
        occu[users_df['Occupation'][idx]] = 1
        tmp = [gender, users_age[idx]]
        tmp.extend(occu)
        users_info[user_id - 1] = tmp

    for idx, movie_id in enumerate(movies_df['movieID']):
        genres = movies_df['Genres'][idx].split('|')
        tmp = np.zeros(all_genres.shape[0])
        for genre in genres:
            tmp[np.where(all_genres == genre)[0][0]] = 1
        movies_info[movie_id - 1] = tmp

    model = build_model(max_user_id, max_movie_id,
                        lat_dim, users_info, movies_info)
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    indices = np.arange(users.shape[0])
    np.random.shuffle(indices)
    val_num = int(users.shape[0] * split_ratio)
    users = users[indices]
    movies = movies[indices]
    ratings = ratings[indices]
    tra_users = users[:-val_num]
    tra_movies = movies[:-val_num]
    tra_ratings = ratings[:-val_num]
    val_users = users[-val_num:]
    val_movies = movies[-val_num:]
    val_ratings = ratings[-val_num:]

    model.fit([tra_users, tra_movies], tra_ratings,
              batch_size=128,
              epochs=40,
              validation_data=([val_users, val_movies], val_ratings))
    model.save(model_path)

    ###########################
    ####        test       ####
    test = pd.read_csv(index4)
    output = model.predict(
        [test['UserID'].values - 1, test['MovieID'].values - 1])
    if args.normal:
        output = output * std + mean
    with open('ans_dnn.csv', 'w') as f:
        f.write("TestDataID,Rating\n")
        for i, rating in enumerate(output):
            r = rating[0]
            if r > 5:
                r = 5
            f.write("{},{}\n".format(i + 1, r))


if __name__ == '__main__':
    main()
