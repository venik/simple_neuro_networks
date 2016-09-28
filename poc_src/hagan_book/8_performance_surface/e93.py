#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ylist = np.linspace(-2.0, 2.0)
xlist = np.linspace(-2.0, 2.0)

X, Y = np.meshgrid(xlist, ylist)
Z = X*X + 2*Y*Y

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z)

plt.grid()
#plt.show()

# i - find minimum along the line
from sympy import Function, hessian, pprint
from sympy.abc import x, y 

#f = Function('f')(x, y)
z = x*x + 2*y*y
pprint(hessian(z, (x, y)))
