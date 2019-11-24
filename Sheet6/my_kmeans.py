
import numpy as np
import scipy, scipy.spatial



def my_kmeans(data, k):
    # TODO:
    # The variable 'data' contains data points in its rows.
    # Initilize 'k' centers randomized.  Afterwards apply Floyd's algorithm
    # until convergence to get a solution of the k-means problem.  Utilize
    # 'scipy.spatial.cKDTree' for nearest neighbor computations.
    # data shape; (262144, 3)
    centers = np.random.random((k, 3)) * 255

    membership_old = None
    while True:
        nn = scipy.spatial.cKDTree(centers)
        distances, membership = nn.query(data)
        if np.all(membership == membership_old):
            break
        membership_old = membership
        for i, center in enumerate(centers):
            currmem = data[membership == i]
            if not currmem.size == 0:
                centers[i] = currmem.mean(axis=0)


    index = membership_old



    return index, centers

