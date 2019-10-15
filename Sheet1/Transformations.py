# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation

v = np.load('Monkey.npy')


# a) limit the ranges of all three axes to [-1,...1]
def plot_models(*models):
    figure = plt.figure()
    ax = mplot3d.Axes3D(figure)
    for m in models:
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(m))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    plt.show()


plot_models(v)

# b) translate and scale the model
t = np.array([1, 0, 0])
vt = 2 * v + t

# show results in a new plot
plot_models(vt)

# c) rotate model around z-axis and display both at the same time
# generate a rotation matrix for z-angle pi/4
r = Rotation.from_euler('xyz', angles=(0, 0, math.pi / 4)).as_dcm()

# transpose the last two axes of v, matrix-vector-multiply (with r^2 for pi/2 rotation), and undo the transposition
vr = np.einsum('ij,...kj->...ki', r @ r, v)
# move model to origin, rotate (like above) and move back to t (= rotate around z-axis at t)
vtr = np.einsum('ij,...kj->...ki', r, vt - t) + t

plot_models(vr, vtr)
