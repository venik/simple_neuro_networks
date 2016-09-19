#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append('../../')
from lib.mnist.mnist import *

# plot estimated data

# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

# import data set
data_set = mnist('../../data_set/mnist', isTrainMode=False)

fd_centroids = open("est_centroids", "r")
npzfile = np.load(fd_centroids)
est_centroids = npzfile['arr_0']
(rows, cols) = est_centroids.shape

fig = plt.figure()
for k in range(1, 11):
    if k < 6:
        line = 1
    else:
        line = 2

    ax = fig.add_subplot(line, 5, k)
    ax.imshow(np.reshape(est_centroids[k-1, :], [28, 28], order='C'), cmap = cm.Greys_r)

plt.show()