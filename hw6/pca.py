#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-22 23:41:19
# @Author  : Jintao Yu (yujintao2013@gmail.com)
# @Link    : https://jintaoyu.github.io

import os
import sys
import numpy as np
import skimage
from skimage import io

# input parameters
# @1: the path of directory
# @2: the reconstructed image
index1 = sys.argv[1]
index2 = sys.argv[2]


class PCA:
    def __init__(self, images_path):
        print("PCA starts ......")
        self.images_path = images_path
        self.images_array = self.get_images()

    def get_images(self):
        files = os.path.join(self.images_path, '*.jpg')
        images_collection = io.ImageCollection(
            files)  # class ImageCollection, len=415
        images_array = images_collection.concatenate().reshape(
            415, -1)  # shape = (415, 1080000)
        return images_array

    def print_specific_face(self, index):
        specific_face = self.images_array[index, :].reshape(600, 600, 3)
        io.imshow(specific_face)
        io.imsave('./specific/face_id={}.jpg'.format(index), specific_face)

    def plot_average_face(self):
        mean_image_array = np.mean(self.images_array, axis=0)
        average_face = np.array(
            mean_image_array, dtype=np.uint8).reshape(600, 600, 3)
        # print(mean_image_array)
        # print(average_face)
        io.imsave('./average/average_face.jpg', average_face)

    def plot_eigen_faces(self, top=10):
        mean_img = np.mean(self.images_array, axis=0)
        # shape = (415, 1080000)
        array_a = self.images_array - mean_img
        # print(new_images_array.shape)

        # U, s, V shapes= (1080000, 415), (415,), (415, 415)
        U, s, V = np.linalg.svd(array_a.T, full_matrices=False)

        for i in range(top):
            eigen_face = -U[:, i].reshape(600, 600, 3)
            M = eigen_face
            M -= np.min(M)
            M /= np.max(M)
            M = (M * 255).astype(np.uint8)
            eigen_face = M
            io.imsave('./eigen/eigen_face_id={}.jpg'.format(i), eigen_face)

    def compute_RMSE(self, arr1, arr2):
        return np.sqrt(np.average((arr1 - arr2) ** 2))

    def reconstruct_faces(self, target_image, top=4):
        target = io.imread(target_image)
        target = target.reshape(1, -1)  # shape = (1, 1080000)

        mean_img = np.mean(self.images_array, axis=0)
        array_a = self.images_array - mean_img

        # U shape = (1080000, 415)
        U, s, V = np.linalg.svd(array_a.T, full_matrices=False)

        target_a = target - mean_img  # shape = (1, 1080000)

        weights = np.dot(target_a, U)

        recon = mean_img + np.dot(weights[:, :top], U[:, :top].T)

        recon -= np.min(recon)
        recon /= np.max(recon)
        recon = (recon * 255).astype(np.uint8)
        recon = recon.reshape(600, 600, 3)

        io.imsave('./reconstruction.jpg'.format(target_image), recon)

    def calculate_ratio(self, top=4):
        mean_img = np.mean(self.images_array, axis=0)
        array_a = self.images_array - mean_img
        U, s, V = np.linalg.svd(array_a.T, full_matrices=False)

        s_sum = np.sum(s)
        print(s_sum)
        for i in range(top):
            print(s[i] / s_sum)


def main():
    pca = PCA(index1)
    # pca.get_images()
    # pca.plot_average_face()
    # pca.plot_eigen_faces()
    pca.reconstruct_faces(index2)
    # pca.calculate_ratio()


if __name__ == '__main__':
    main()
