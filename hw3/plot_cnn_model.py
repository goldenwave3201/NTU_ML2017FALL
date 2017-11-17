#!/usr/bin/env python3
# -- coding: utf-8 --

import pydot
import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model


def main():
    parser = argparse.ArgumentParser(prog='plot_cnn_model.py',
                                     description='Plot the cnn model.')
    parser.add_argument('--model', type=str, default='./cnn_model')
    args = parser.parse_args()

    emotion_classifier = load_model(args.model)
    emotion_classifier.summary()
    plot_model(emotion_classifier, to_file='./image/cnn_model.png')


if __name__ == '__main__':
    main()
