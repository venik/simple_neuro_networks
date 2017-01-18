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

# import data set
data_set = mnist('../../data_set/mnist')
frame_num = 1000
row_size = 784

input = np.zeros((row_size, frame_num))
for frame in range(1, frame_num):
    (_, data) = data_set.get_next_frame()
    input[:, frame] = data

# input = np.array((1, 2, 3, 4)).reshape(2, 2)
# print("input: " + str(input))

# print("input:" + str(input.shape))
mean = input.mean(axis = 1)
input = input - mean.reshape(row_size, 1)
input = input.transpose() / np.sqrt(input.shape[1] - 1)
U, s, V = np.linalg.svd(input, full_matrices = False)

# print("mean: " + str(mean))
# print("input: " + str(input))
# print("U: " + str(U))
# print("singular: " + str(s))

# s_normilized = s / np.sum(s)
s_normilized = s
#
# print("singular values: " + str(s_normilized))
#
# # range of interest
short_range = 250
plt.plot(np.arange(1, short_range), s_normilized[1:short_range])
plt.grid()
plt.show()

# test - plot
# img = input[:, 1]
# array = np.reshape(data, [28, 28], order='C')
#plt.imshow(array, cmap = cm.Greys_r)
#plt.show()