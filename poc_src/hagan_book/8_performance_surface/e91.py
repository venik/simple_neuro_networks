#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# i - sketch a contour plot
ylist = np.linspace(-1.0, 1.0)
xlist = np.linspace(-1.0, 1.0)

X, Y = np.meshgrid(xlist, ylist)
Z = X*(3*X-Y)-X+Y*(-X + 3*Y)-Y + 2

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z)

plt.grid()
plt.show()

# iii - perform two iterations
# v - what is the maximum stable learning rate
# vi - write a matlab m-file to proove i-v (implemented on Python)
x0 = np.array((0, 0)).reshape(2, 1)
A = np.array((6, -2, -2, 6)).reshape(2, 2)
free_vec = np.array((-1, -1)).reshape(2, 1)
alpha = 0.1
# Maximum rate for [0, 0] is 0.25 - one step to the global minimum
#alpha = 0.25

for i in range(0, 10):
   x = x0 - alpha * (A.dot(x0) + free_vec)
   x0 = x
   print("x: " + str(x))

