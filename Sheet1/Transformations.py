# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

v = np.load('Monkey.npy')

figure=plt.figure()
ax = mplot3d.Axes3D(figure)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(v))

# a) limit the ranges of all three axes to [-1,...1]
# ...


# b) translate and scale the model
# ...

# show results in a new plot
# ...


# c) rotate model around z-axis and display both at the same time
# ...


plt.show()