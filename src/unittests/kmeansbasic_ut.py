#!/usr/bin/env python
import numpy as np
import unittest

import sys
sys.path.append('../../')
from lib.kmeansbasic import KMeansBasic
from lib.dataset.dataset import DataSet

# K-Means basic against MNIST UnitTests
# GPLv2 License

__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'


class KMeansTestDataset(DataSet):
    _data_set = None
    _data_index = 0
    _rows = 0
    _cols = 0

    def __init__(self):
        self._data_set = np.array((1, 1, 3, 1, 2, 2.5, 7, 3, 8, 4, 9, 2)).reshape(6, 2)
        self._index = 0
        (self._rows, self._cols) = self._data_set.shape
        # print("shape: " + str(self._data_set.shape))
        # print("dataset:\n " + str(self._data_set))


    def get_next_frame(self):
        data = self._data_set[self._index, :]
        self._index += 1
        return 0, data

    def seek(self, pos):
        self._index = pos

    def get_num_of_frames_in_dataset(self):
        return self._rows


class TestOneLayerPerceptron(unittest.TestCase):

    def setUp(self):
        pass

    def testPositiveOutputWithBias(self):
        data_set = KMeansTestDataset()
        (_, m0) = data_set.get_next_frame()
        (_, m1) = data_set.get_next_frame()

        centers = np.array((m0, m1)).reshape(2, 2)

        data_set.seek(0)
        kmeans = KMeansBasic(data_set)
        (res, centroids) = kmeans.recalculate_centroids(iterations=10, centroids=centers)
        self.assertTrue(res, True)
        self.assertTrue((centroids == np.array((2, 1.5, 8, 3)).reshape(2, 2)).all())

if __name__ == '__main__':
    unittest.main()
