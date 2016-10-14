#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# i - sketch a contour plot
ylist = np.linspace(-3, 3)
xlist = np.linspace(-3, 3)

X, Y = np.meshgrid(xlist, ylist)
Z = 3.5*X*X - 9*X*Y - 8.5*Y*Y + 16*X + 8*Y

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z, 10)

from sympy import Function, hessian, pprint
from sympy.abc import x, y 

z = 3.5*x*x - 9*x*y - 8.5*y*y + 16*x + 8*y
pprint(hessian(z, (x, y)))
pprint(hessian(z, (x, y)).eigenvals())

# eigenvals are -20 and 10, so alpha < 2/10 < 0.2
iterations = 20
alpha = 0.01
solution = np.zeros((iterations, 2))
x0 = np.array((1, 1)).reshape(2, 1)
solution[0, :] = x0.reshape(1, 2)
print("x0: " + str(solution[0, :]))
for i in range(1, iterations):
    g = np.array((-7, -6, -6, 2)).reshape(2, 2).dot(x0)
    x = x0 - alpha * g

    x0 = x
    solution[i, :] = x.reshape(1, 2)
    print("x: " + str(solution[i, :]))

cc.plot(solution[:, 0], solution[:, 1], 'r^')

plt.grid()
plt.show()
