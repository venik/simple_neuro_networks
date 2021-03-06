#!/usr/bin/env python

# PCA against MNIST, trying to get reasonable number of classes,
# data set has 10 classes - from 0 to 9
# FIXME: doesnt work now - finds to many classes
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt


def load_mnist():
    # import data set
    data_set = mnist('../../data_set/mnist')

    frame_num = 1000
    row_size = data_set._frame_size

    data_matrix = np.zeros((row_size, frame_num))
    for frame in range(1, frame_num):
        (_, data) = data_set.get_next_frame()
        data_matrix[:, frame] = data

    return data_matrix, row_size


def load_fake_data():
    data_matrix = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9)).reshape(3, 3)

    return data_matrix, 3


def pca_svd(data_matrix, row_size):
    # Whitening
    # http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
    mean = data_matrix.mean(axis=1)
    data_matrix = data_matrix - mean.reshape(row_size, 1)
    data_matrix = data_matrix.transpose() / np.sqrt(data_matrix.shape[1] - 1)

    U, s, V = np.linalg.svd(data_matrix, full_matrices=False)

    return U, s, V

(data_matrix, row_size) = load_mnist()
(U, s, V) = pca_svd(data_matrix, row_size)

sum50 = np.sum(s[0:49])
sumAll = np.sum(s)

print("sum50 / sumAll = {:.2f}".format(sum50/sumAll))

#print("input: \n" + str(input))

# print("input:" + str(input.shape))

#print("mean: " + str(mean))
#print("input: \n" + str(input))
# print("U: " + str(U))
# print("singular: " + str(s))

# plot only for 10+ eigenvalues
if (row_size >= 10):
    # range of interest
    short_range = 250
    plt.plot(np.arange(1, short_range), s[1:short_range])
    plt.xlabel('factor number')
    plt.ylabel('eigenvalue')
    plt.grid()
    plt.show()


# test - plot MNIST
# img = input[:, 1]
# array = np.reshape(data, [28, 28], order='C')
#plt.imshow(array, cmap = cm.Greys_r)
#plt.show()
