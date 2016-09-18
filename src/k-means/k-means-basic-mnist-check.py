#!/usr/bin/env python
import sys
sys.path.append('../../')
from lib.mnist.mnist import *
from array import array
from sys import maxint
from numpy import linalg as la

# 1 layer perceptron neuronetwork against MNIST
# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

# import data set
data_set = mnist('../../data_set/mnist', isTrainMode=False)

fd_centroids = open("est_centroids", "r")
npzfile = np.load(fd_centroids)
est_centroids = npzfile['arr_0']
(rows, cols) = est_centroids.shape

samples_in_cluster = array("d", (0.0 for _ in range(0, cols)))
sum_in_cluster = array("d", (0.0 for _ in range(0, cols)))

for k in range(0, data_set.get_num_of_frames_in_dataset()):
    (label, data) = data_set.get_next_frame()

    # calculate distance to a closest centroid
    closest_cluster = 0
    min_distance = maxint
    for cluster in range(0, rows):
        res = la.norm(data - est_centroids[cluster, :])
        if res <= min_distance:
            min_distance = res
            closest_cluster = cluster

    sum_in_cluster[closest_cluster] += label
    samples_in_cluster[closest_cluster] += 1

for k in range(0, rows):
    sum_in_cluster[k] /= samples_in_cluster[k]
    print(str(k) + ": round:" + str(round(sum_in_cluster[k])) + " average sum: {:.2f}".format(sum_in_cluster[k]))

# rounded_res = [round(k) for k in sum_in_cluster]
# print("list: " + str(sorted(rounded_res)))

# tear down
data_set.tear_down()
fd_centroids.close()
