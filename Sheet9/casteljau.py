import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull

points = np.array([[ 2, 11],
                   [ 4, 2 ],
                   [ 8, 9 ],
                   [10, 1 ],
                   [18, 3 ],
                   [19, 11]])

t_samples = np.arange(0, 1, 0.001)

def bezierCurve(t, n):
    return deCasteljau(t, n, n)

def deCasteljau(t, i, n):
    print('i: ' + str(i) + ' n: ' + str(n))
    # TODO

def controlCurve(t, points):
    n = points.shape[0]
    # TODO

# TODO
# do plotting 

