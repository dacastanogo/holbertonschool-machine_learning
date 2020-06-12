#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images2 = np.array([[[3, 0, 1, 2, 7, 4], [1, 5, 8, 9, 3, 1], [2, 7, 2, 5, 1, 3], [0, 1, 3, 1, 7, 8], [4, 2, 1, 6, 2, 8], [2, 4, 5, 2, 3, 9]]])
    images3 = np.array([[[10, 10, 10, 0, 0, 0], [10, 10, 10, 0, 0, 0], [10, 10, 10, 0, 0, 0], [10, 10, 10, 0, 0, 0], [10, 10, 10, 0, 0, 0], [10, 10, 10, 0, 0, 0]]])
    print(images2.shape)
    print(images2)
    images_conv = convolve_grayscale_same(images2, kernel)
    print(images_conv.shape)
    print(images_conv)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
