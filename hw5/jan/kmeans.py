##
%matplotlib tk
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def kmeans(k, X):
    assert(k >= 1)

    mu = X[np.random.choice(X.shape[0], k, replace=False), :]
    it = 0

    prevIndices = -np.ones(X.shape[0])
    while True:
        distances = np.linalg.norm(mu[0,:]-X, axis=1)
        indices = np.zeros(X.shape[0])
        for i in range(1,k):
            tempDistances = np.linalg.norm(mu[i,:]-X, axis=1)
            shorter = tempDistances < distances
            distances[shorter] = tempDistances[shorter]
            indices[shorter] = i
        if it <= 1:
            assignmentsTwoIterations = indices

        for i in range(k):
            assigned = indices == i
            mu[i,:] = sum(X[assigned])/np.count_nonzero(assigned)

        if (indices == prevIndices).all():
            changedAssignment = indices != assignmentsTwoIterations
            plt.plot(X[changedAssignment, 0], X[changedAssignment, 1], 'o',
                     label='Has changed assignment', markerfacecolor='none',
                     markersize=12)
            for i in range(k):
                assigned = indices == i
                plt.plot(mu[i,0], mu[i,1], 'o', label='Cluster #{} mean'.format(i+1))
                plt.plot(X[assigned, 0], X[assigned, 1], '.',
                         label='Cluster #{}'.format(i+1))
            plt.legend()
            return
        prevIndices = indices
        it += 1



mat = scipy.io.loadmat('hw5_p1a.mat')
X = mat['X']
kmeans(2, X)
##
