#!/usr/bin/env python

import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from array import array
from numpy import linalg as la

# 1 layer perceptron neuronetwork against MNIST
# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

# import data set
data_set = mnist('../../data_set/mnist', isTrainMode=False)

fd_centroids = open("est_centroids", "rb")
npzfile = np.load(fd_centroids)
est_centroids = npzfile['est_centroids']
(rows, cols) = est_centroids.shape

samples_in_cluster = array("d", (0.0 for _ in range(0, cols)))
sum_in_cluster = array("d", (0.0 for _ in range(0, cols)))

hits = 0.0
num_of_samples = data_set.get_num_of_frames_in_dataset()
for k in range(0, num_of_samples):
    (label, data) = data_set.get_next_frame()

    # calculate distance to a closest centroid
    closest_cluster = 0
    min_distance = sys.maxsize
    for cluster in range(0, rows):
        res = la.norm(data - est_centroids[cluster, :])
        if res < min_distance:
            min_distance = res
            closest_cluster = cluster

    # print("cluster: " + str(closest_cluster) + " label: " + str(label))
    if closest_cluster == label:
        hits += 1

    samples_in_cluster[closest_cluster] += 1

# hits /= data_set.get_num_of_frames_in_dataset()
print("hits {:.2f} out of {:d} samples ({:.2f}%)".format(hits, num_of_samples, hits * 100 / num_of_samples))

# tear down
data_set.tear_down()
fd_centroids.close()
