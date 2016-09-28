#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.4 ii

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# i - sketch a contour plot
ylist = np.linspace(-3, 1.2)
xlist = np.linspace(-3, 1.2)

X, Y = np.meshgrid(xlist, ylist)
Z = 5*X*X - 6*X*Y + 5*Y*Y + 4*X + 4*Y

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z, 10)

from sympy import Function, hessian, pprint
from sympy.abc import x, y 

z = 5*x*x - 6*x*y + 5*y*y + 4*x + 4*y
pprint(hessian(z, (x, y)))
pprint(hessian(z, (x, y)).eigenvals())

# eigenvals are 4 and 16, so alpha < 2/16 < 0.25
alpha = 0.20
solution = np.zeros((10, 2))
x0 = np.array((1, 1)).reshape(2, 1)
solution[0, :] = x0.reshape(1, 2)
print("x0: " + str(solution[0, :]))
for i in range(1, 10):
    g = np.array((10, -6, -6, 10)).reshape(2, 2).dot(x0) + np.array((4, 4)).reshape(2, 1)
    x = x0 - alpha * g

    x0 = x
    solution[i, :] = x.reshape(1, 2)
    print("x: " + str(solution[i, :]))

cc.plot(solution[:, 0], solution[:, 1], 'r^')

plt.grid()
plt.show()
