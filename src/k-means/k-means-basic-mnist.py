#!/usr/bin/env python

# 1 layer perceptron neuro network against MNIST
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from lib.kmeansbasic import KMeansBasic

def get_first_centroids(dataset):
    (_, m0) = dataset.get_arbitrary_frame(0)
    (_, m1) = dataset.get_next_frame()
    (_, m2) = dataset.get_next_frame()
    (_, m3) = dataset.get_next_frame()
    (_, m4) = dataset.get_next_frame()
    (_, m5) = dataset.get_next_frame()
    (_, m6) = dataset.get_next_frame()
    (_, m7) = dataset.get_next_frame()
    (_, m8) = dataset.get_next_frame()
    (_, m9) = dataset.get_next_frame()

    return np.array((m0, m1, m2, m3, m4, m5, m6, m7, m8, m9)).reshape(10, 784)


def get_centroids_around_digits(dataset):
    center = 0
    data_centroids = np.zeros(784 * 10).reshape(10, 784)

    # reset to 0 frame
    dataset.get_arbitrary_frame(100)
    while center < 10:
        (label, data) = dataset.get_next_frame()
        if label == center:
            data_centroids[center, :] = data
            center += 1

    return data_centroids

# import data set
data_set = mnist('../../data_set/mnist')

kmeans = KMeansBasic(data_set)

# centroids = get_first_centroids(data_set)
centroids = get_centroids_around_digits(data_set)

(_, est_centroids) = kmeans.recalculate_centroids(iterations=20, centroids=centroids)

# save weights and biases
fd_neuro = open("est_centroids", "w")
np.savez(fd_neuro, est_centroids=est_centroids, centroids=centroids)

# tear down
data_set.tear_down()
fd_neuro.close()
