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
Z = 7/2 * X*X - 6*X*Y - Y*Y

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z)

#plt.grid()
#plt.show()

from sympy import Function, hessian, pprint
from sympy.abc import x, y 

z = (7/2) * x*x - 6*x*y - y*y
pprint(hessian(z, (x, y)))
pprint(hessian(z, (x, y)).eigenvals())

x0 = np.array((1, 1)).reshape(2, 1)
alpha = 0.1
for i in range(0, 10):
    g = np.array((7, -6, -6, -2)).reshape(2, 2).dot(x0)
    x = x0 - alpha * g

    x0 = x
    print("x: " + str(x))
