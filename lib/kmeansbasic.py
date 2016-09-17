__author__ = 'Sasha Nikiforov nikiforov.al[a]gmail.com'
__copyright__ = "Copyright 2016, Sasha Nikiforov"

__license__ = "GPLv2"
__maintainer__ = "Sasha Nikiforov"
__email__ = "nikiforov.al[a]gmail.com"
__status__ = "Development"

from sys import maxint, getsizeof
from numpy import linalg as la
import numpy as np
from array import array

class KMeansBasic(object):
    _data_set = None
    _data_set_list = None
    _data_set_list_size = 0

    def __init__(self, data_set):
        self._data_set = data_set
        self._data_set_list = []
        for sample in range(0, 50000):
            self._data_set_list.append(self._data_set.get_next_frame())

        self._data_set_list_size = sample
        print("str: " + str(len(self._data_set_list)) + " size: " + str(getsizeof(self._data_set_list)))

        return

    def recalculate_centroids(self, iterations, centroids):
        (rows, cols) = centroids.shape

        for iter in range(0, iterations):
            est_centers = np.zeros((rows, cols))
            samples_in_cluster = array("d", (0.0 for _ in range(0, cols)))
            for (_, data) in self._data_set_list:

                # calculate distance to a closest centroid
                closest_cluster = 0
                min_distance = maxint
                for cluster in range(0, cols):
                    res = la.norm(data - centroids[:, cluster])
                    if res <= min_distance:
                        min_distance = res
                        closest_cluster = cluster

                    #print(str(cluster) + ": norm: " + str(res))

                samples_in_cluster[closest_cluster] += 1
                np.add(est_centers[:, closest_cluster], data, est_centers[:, closest_cluster])
                # print("== Closest cluster: " + str(closest_cluster))

            # normalize coordinates
            for center in range(0, cols):
                # print(str(center) + ": norm " + str(la.norm(est_centers[:, center])))
                est_centers[:, center] /= samples_in_cluster[center]
                # print(str(center) + ": norm " + str(la.norm(est_centers[:, center])))

            # update centroids
            centroids = est_centers

            summ = 0.0
            for k in samples_in_cluster:
                summ += k

            print("sum:" + str(summ) + " samples in cluster: " + str(samples_in_cluster))

        return False, centroids
