#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# matrix fatorization with the bias

import sys
import pandas as pd
import numpy as np
import keras
from keras.models import Model, load_model

# test.csv
index1 = sys.argv[1]
# predict file
index2 = sys.argv[2]
# movie
index3 = sys.argv[3]
# user
index4 = sys.argv[4]

model_path = './model/mf_bias.model'


def main():
    rating_mean = 3.581712
    rating_std = 1.116898

    test = pd.read_csv(index1)
    model = load_model(model_path)

    output = model.predict(
        [test['UserID'].values - 1, test['MovieID'].values - 1])
    output = output * rating_std + rating_mean
    with open(index2, 'w') as f:
        f.write("TestDataID,Rating\n")
        for i, rating in enumerate(output):
            r = rating[0]
            if r > 5:
                r = 5
            f.write("{},{}\n".format(i + 1, r))


if __name__ == '__main__':
    main()
