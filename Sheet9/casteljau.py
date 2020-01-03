import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull

points = np.array([[2, 11],
                   [4, 2],
                   [8, 9],
                   [10, 1],
                   [18, 3],
                   [19, 11]])

t_samples = np.arange(0, 1, 0.001)


def bezierCurve(t, n):
    return deCasteljau(t, 0, n)


def deCasteljau(t, i, n):
    print('i: ' + str(i) + ' n: ' + str(n))
    if n == 0:
        return points[i]
    return t * deCasteljau(t, i + 1, n - 1) + (1 - t) * deCasteljau(t, i, n - 1)


def controlCurve(t, points):
    n = points.shape[0]
    left = np.floor(t * (n - 1)).astype(int)
    right = left + 1
    frac = t * (n - 1) - left
    return (1 - frac) * points[left] + frac * points[right]


# do plotting
n = points.shape[0]
control = np.stack([controlCurve(t, points) for t in t_samples])
bezier = np.stack([bezierCurve(t, n - 1) for t in t_samples])

plt.plot(control[:, 0], control[:, 1], c='red', label="control")
plt.scatter(points[:, 0], points[:, 1], marker='s', c='red')
plt.plot(bezier[:, 0], bezier[:, 1], label="bézier")
plt.legend()
plt.show()
