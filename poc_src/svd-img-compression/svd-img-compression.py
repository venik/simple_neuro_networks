#!/bin/env python3
from scipy.ndimage.io import imread
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

img = imread('./lenin.jpg', flatten=True, mode=None)
U, s, V = np.linalg.svd(img, full_matrices=False)

rank = np.linalg.matrix_rank(img, tol=None)
approx_rank = 25 
compression = approx_rank * (img.shape[0] + img.shape[1]) / (img.shape[0] * img.shape[1])

real_img = np.dot(U, np.dot(np.diag(s), V))
S = np.diag(s)
approx_img = np.dot(
    U[:, 0:approx_rank],
        np.dot(
            S[0:approx_rank, 0:approx_rank],
            V[0:approx_rank, :]))

plt.subplot(1, 2, 1)
plt.imshow(real_img, cmap=cm.Greys_r)
plt.xlabel('Real image rank: ' + str(rank))

plt.subplot(1, 2, 2)
plt.imshow(approx_img, cmap=cm.Greys_r)
plt.xlabel('Approx rank: ' + str(approx_rank) + ' compression: {:2.4f}'.format(compression))

plt.show()
