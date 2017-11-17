#!/usr/local/bin/env python3
# -- coding: utf-8 --
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.layers.noise import GaussianNoise


def read_features(filename):
    col0 = []
    col1 = []
    with open(filename, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            fields = line.strip().split(',')
            col0.append(int(fields[0]))
            col1.append(list(map(int, fields[1].split())))
            if idx == 300:
                break
    return np.array(col0), np.array(col1)


def main():
    data_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load model
    model = load_model(model_path)
    print('Model loaded')

    layer_dict = dict([layer.name, layer] for layer in model.layers[0:])
    print('layer_dict =', layer_dict)

    input_img = model.input
    name_ls = ['conv2d_1', 'leaky_re_lu_1', 'conv2d_2', 'leaky_re_lu_2', 'max_pooling2d_1', 'dropout_1', 'conv2d_3', 'leaky_re_lu_3', 'conv2d_4',
               'leaky_re_lu_4', 'max_pooling2d_2', 'dropout_2', 'conv2d_5', 'leaky_re_lu_5', 'conv2d_6', 'leaky_re_lu_6', 'max_pooling2d_3', 'dropout_3']
    collect_layers = [K.function([input_img, K.learning_phase()], [
                                 layer_dict[name].output]) for name in name_ls]

    # Load data
    labels, pixels = read_features(data_path)
    choose_id = 9
    photo = pixels[choose_id].reshape(1, 48, 48, 1)

    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0])
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        nb_filter = 32
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter / 8, 8, i + 1)
            ax.imshow(im[0][0, :, :, i], cmap='summer')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer: {} (Given image {})'.format(
            name_ls[cnt], choose_id))
        fig.savefig('layer_{}.png'.format(name_ls[cnt]))


if __name__ == "__main__":
    main()
