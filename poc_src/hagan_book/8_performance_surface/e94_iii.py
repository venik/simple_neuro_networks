#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.4 iii

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# i - sketch a contour plot
ylist = np.linspace(-1, 1)
xlist = np.linspace(-1, 1)

X, Y = np.meshgrid(xlist, ylist)
Z = 9/2*X*X - 2*X*Y + 3*Y*Y + 2*X - Y

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z, 10)

from sympy import Function, hessian, pprint
from sympy.abc import x, y 

z = 4.5*x*x - 2*x*y + 3*y*y + 2*x - y
pprint(hessian(z, (x, y)))
pprint(hessian(z, (x, y)).eigenvals())

# eigenvals are -5 and 10, so alpha < 2/10 = 0.2
alpha = 0.1 
solution = np.zeros((10, 2))
x0 = np.array((1, 1)).reshape(2, 1)
solution[0, :] = x0.reshape(1, 2)
print("x0: " + str(solution[0, :]))
for i in range(1, 10):
    g = np.array((9, -2, -2, 6)).reshape(2, 2).dot(x0) + np.array((2, -1)).reshape(2, 1)
    x = x0 - alpha * g

    x0 = x
    solution[i, :] = x.reshape(1, 2)
    print("x: " + str(solution[i, :]))

cc.plot(solution[:, 0], solution[:, 1], '-r^')

plt.grid()
plt.show()
