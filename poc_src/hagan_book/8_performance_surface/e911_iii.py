#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.11 iii

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# i - sketch a contour plot
ylist = np.linspace(-1, 1)
xlist = np.linspace(-1, 1)

X, Y = np.meshgrid(xlist, ylist)
Z = 4.5*X*X - 2*X*Y + 3*Y*Y + 2*X - Y

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

# first approximationa and 2 steps
iterations = 5
solution = np.zeros((iterations, 2))
x0 = np.array((1., 1.)).reshape(2, 1)
solution[0, :] = x0.reshape(1, 2)

# step 0
p0 = np.array(( 9*x0[0] - 2*x0[1] + 2, 6*x0[1] - 2*x0[0] -1)).reshape(2, 1)
A0 = np.array(( 9, -2, -2, 6 )).reshape(2, 2)
numerator0 = p0.reshape(1,2).dot(p0)
denumerator0 = p0.reshape(1,2).dot(A0).dot(p0)
alpha0 = np.divide(numerator0, denumerator0)

x1 = x0 - alpha0 * p0
print("x1 = " + str(x1))

solution[1, :] = x1.reshape(1, 2)

for i in range(2, iterations):
    # step 1
    g1 = np.array(( 9*x1[0] - 2*x1[1] + 2, 6*x1[1] - 2*x1[0] -1)).reshape(2, 1)
    A1 = np.array(( 9, -2, -2, 6 )).reshape(2, 2)

    beta1 = np.divide(g1.reshape(1, 2).dot(g1), p0.reshape(1, 2).dot(p0))
    p1 = -g1 + beta1 * p0

    numerator1 = g1.reshape(1,2).dot(p1)
    denumerator1 = p1.reshape(1,2).dot(A1).dot(p1)
    alpha1 = np.divide(numerator1, denumerator1)

    x2 = x1 - alpha1 * p1
    solution[i, :] = x2.reshape(1, 2)
    print("x2 = " + str(x2))

    x1 = x2

# =================
cc.plot(solution[:, 0], solution[:, 1], '-r^')

plt.grid()
plt.show()
