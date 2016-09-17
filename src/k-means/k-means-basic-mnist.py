#!/usr/bin/env python

# 1 layer perceptron neuro network against MNIST
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from lib.kmeansbasic import KMeansBasic

# import data set
data_set = mnist('../../data_set/mnist')

# Dimensions size, MNIST is 784 (28 by 28 picture)
dim = 784
kmeans = KMeansBasic(data_set)

# centroids = np.array()
(_, m0) = data_set.get_next_frame()
(_, m1) = data_set.get_next_frame()
(_, m2) = data_set.get_next_frame()
(_, m3) = data_set.get_next_frame()
(_, m4) = data_set.get_next_frame()
(_, m5) = data_set.get_next_frame()
(_, m6) = data_set.get_next_frame()
(_, m7) = data_set.get_next_frame()
(_, m8) = data_set.get_next_frame()
(_, m9) = data_set.get_next_frame()

centroids = np.array([[m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]]).reshape(784, 10)

kmeans.recalculate_centroids(iterations=20, centroids=centroids)

# tear down
data_set.tear_down()