#!/usr/local/bin/env python3
# -- coding: utf-8 --
import sys
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import exposure


def read_features(filename):
    col0 = []
    col1 = []
    with open(filename, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            fields = line.strip().split(',')
            col0.append(int(fields[0]))
            col1.append(list(map(int, fields[1].split())))
            if idx == 100:
                break
    return np.array(col0), np.array(col1)


def main():
    data_path = sys.argv[1]
    model_path = sys.argv[2]

    # load the model
    model = load_model(model_path)
    print('Loaded model from {}'.format(model_path))

    # load the training data
    labels, pixels = read_features(data_path)

    input_img = model.input
    img_ids = [3, 5, 9]

    for idx in img_ids:
        val_proba = model.predict(pixels[idx].reshape(1, 48, 48, 1) / 255.)
        pred = val_proba.argmax(axis=-1)
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        print('idx = {}'.format(idx))
        print('  ==> label =', labels[idx])
        print('  ==> val_proba =', val_proba)
        print('  ==> pred =', pred)

        # set learning_phase = 0 for test
        grad_val = fn([pixels[idx].reshape(1, 48, 48, 1) / 255., 0])[0]
        grad_val = grad_val.reshape(48, 48)
        grad_val = np.abs(grad_val)
        grad_val = (grad_val - np.mean(grad_val)) / (np.std(grad_val) + 1e-10)
        grad_val *= 0.1
        grad_val += 0.5
        grad_val = np.clip(grad_val, 0, 1)
        grad_val /= np.max(grad_val)

        heatmap = grad_val

        thres = 0.55

        plt.figure()
        plt.imshow(pixels[idx].reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('origin-{}.png'.format(idx), dpi=100)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('heatmap-{}.png'.format(idx), dpi=100)

        see = pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('partial-{}.png'.format(idx), dpi=100)


if __name__ == "__main__":
    main()
