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
est_centroids = npzfile['est_centroids']
centroids = npzfile['centroids']

(rows, cols) = est_centroids.shape

ax = None
fig1 = plt.figure(1)
for k in range(1, 11):
    if k == 4:
        ax.set_title('Estimated\ncetroids')

    ax = fig1.add_subplot(4, 5, k)
    ax.imshow(np.reshape(est_centroids[k-1, :], [28, 28], order='C'), cmap=cm.Greys_r, aspect='auto')
    plt.axis('off')

for k in range(11, 21):
    if k == 14:
        ax.set_title('First guess')

    ax = fig1.add_subplot(4, 5, k)
    ax.imshow(np.reshape(centroids[k -11, :], [28, 28], order='C'), cmap=cm.Greys_r, aspect='auto')
    plt.axis('off')

plt.tight_layout()
plt.show()
