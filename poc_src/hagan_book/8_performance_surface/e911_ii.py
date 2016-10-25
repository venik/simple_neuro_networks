#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.11 ii

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

# first approximationa and 2 steps
solution = np.zeros((2, 2))
x0 = np.array((1., 1.)).reshape(2, 1)
solution[0, :] = x0.reshape(1, 2)

# step 0
p0 = np.array(( 10*x0[0] - 6*x0[1] + 4, 10*x0[1] - 6*x0[0] + 4)).reshape(2, 1)
A0 = np.array((10, -6, -6, 10)).reshape(2, 2)
numerator0 = p0.reshape(1,2).dot(p0)
denumerator0 = p0.reshape(1,2).dot(A0).dot(p0)
alpha0 = np.divide(numerator0, denumerator0)

x1 = x0 - alpha0 * p0
print("x1 = " + str(x1))

solution[1, :] = x1.reshape(1, 2)

cc.plot(solution[:, 0], solution[:, 1], 'r^')

plt.grid()
plt.show()
