import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

SIGMA = 10
RHO = 28
BETA = 8 / 3


def LorenzAttractor(t, X):
    """
    Computes the force at a time t and a position X
    t: the current time. In this case, the function is constant in t
    X: the position as [x, y, z] array
    """
    x = np.zeros((3,))
    # your code goes here...
    x[0] = SIGMA * (X[1] - X[0])
    x[1] = X[0] * (RHO - X[2]) - X[1]
    x[2] = X[0] * X[1] - BETA * X[2]
    return x


def Euler(f, y0, stepsize, steps):
    """
    Compute positions using the Euler integration scheme.
    f:			The function that is being integrated
    y0:			The start position
    stepsize:	Delta for each step
    steps:		Amount of integration steps
    """
    x = np.zeros((3, steps))
    # your code goes here...
    yi = y0
    ti = 0
    for i in range(steps):
        yi += stepsize * f(ti, yi)
        ti += stepsize
        x[:, i] = yi
    return x


def RungeKutta4thOrder(f, y0, stepsize, steps):
    """
    Compute positions using the Runge-Kutta 4th Order integration scheme.
    f:			The function that is being integrated
    y0:			The start position
    stepsize:	Delta for each step
    steps:		Amount of integration steps
    """
    x = np.zeros((3, steps))
    # your code goes here...
    yi = y0
    ti = 0
    k = np.zeros((4, 3))
    for i in range(steps):
        k[0] = f(ti, yi)
        k[1] = 2 * f(ti + stepsize/2, yi + 0.5 * stepsize * k[0])
        k[2] = 2 * f(ti + stepsize/2, yi + 0.5 * stepsize * k[1])
        k[3] = f(ti + stepsize, yi + stepsize * k[2])
        yi += stepsize * (1 / 6) * k.sum(axis=0)
        ti += stepsize
        x[:, i] = yi
    return x


# Compute Values
y0 = [-1, 3, 4]
euler = Euler(LorenzAttractor, y0, 0.025, 3000)
rungekutta = RungeKutta4thOrder(LorenzAttractor, y0, 0.025, 3000)
ref = odeint(lambda y, t: LorenzAttractor(t, y), y0, np.arange(0, 0.025 * 3000, 0.025)).T

# Plot Everything
plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Euler Integration")
ax.plot(euler[0, :], euler[1, :], euler[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Runge Kutta Integration")
ax.plot(rungekutta[0, :], rungekutta[1, :], rungekutta[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("ODE Integrate")
ax.plot(ref[0, :], ref[1, :], ref[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Comparison")
ax.plot(euler[0, :], euler[1, :], euler[2, :], label="Euler")
ax.plot(rungekutta[0, :], rungekutta[1, :], rungekutta[2, :], label="Runge Kutta")
ax.plot(ref[0, :], ref[1, :], ref[2, :], label="ODE Integrate")
plt.legend()

plt.show()
