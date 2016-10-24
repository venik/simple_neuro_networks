#!/bin/env python

# Sasha Nikiforov
# [NND] excersize E9.10

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ylist = np.linspace(-5, 5)
xlist = np.linspace(-5, 5)

X, Y = np.meshgrid(xlist, ylist)
Z = (X + Y)**4 - 12*X*Y + X + Y + 1

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z)

cc = fig.add_subplot(122)
cc.contour(X, Y, Z, 10)

from sympy import Function, hessian, pprint
from sympy.abc import x, y 

z = (x + y)**4 - 12*x*y + x + y + 1
pprint(hessian(z, (x, y)))
pprint(hessian(z, (x, y)).eigenvals())

iterations = 20
alpha = 0.005
solution_n = np.zeros((iterations, 2))
solution_s = np.zeros((iterations, 2))

# First approximation of the optimal point (test here!)
x0_n = x0_s = np.array((-1, -2)).reshape(2, 1)
solution_n[0, :] = x0_n.reshape(1, 2)
solution_s[0, :] = x0_s.reshape(1, 2)

for i in range(1, iterations):
    # Newton method
    A = np.array(( 12*(x0_n[0] + x0_n[1])**2, 12*(x0_n[0] + x0_n[1])**2 - 12, 12*(x0_n[0] + x0_n[1])**2 - 12, 12*(x0_n[0] + x0_n[1])**2)).reshape(2, 2)
    g_n = np.array(( 4*(x0_n[0] + x0_n[1])**3 - 12*x0_n[1] + 1,  4*(x0_n[0] + x0_n[1])**3 - 12*x0_n[0] + 1)).reshape(2, 1)

    x_n = x0_n - np.linalg.inv(A).dot(g_n)

    x0_n = x_n
    solution_n[i, :] = x_n.reshape(1, 2)
    print("x_n: " + str(solution_n[i, :]))

    # steepest descent
    g_s = np.array(( 4*(x0_s[0] + x0_s[1])**3 - 12*x0_s[1] + 1,  4*(x0_s[0] + x0_s[1])**3 - 12*x0_s[0] + 1)).reshape(2, 1)
    x_s = x0_s - alpha * g_s

    x0_s = x_s
    solution_s[i, :] = x_s.reshape(1, 2)
    print("x_s: " + str(solution_s[i, :]))

cc.plot(solution_n[:, 0], solution_n[:, 1], 'r^')
cc.plot(solution_s[:, 0], solution_s[:, 1], 'g^')

plt.grid()
plt.show()
