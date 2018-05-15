##
%matplotlib tk
import scipy.io
from scipy.spatial.distance import squareform, pdist
import numpy as np
import matplotlib.pyplot as plt

def rbfkernel(X, sigma):
    P = squareform(pdist(X))
    return np.exp(-P**2/(2*sigma**2))

def getDistances(kernelMatrix, row, assigned):
    nAssigned = np.count_nonzero(assigned)
    a = kernelMatrix[row, row]
    b = 2*np.sum(kernelMatrix[row, assigned])/nAssigned
    c = np.sum(kernelMatrix[np.ix_(assigned, assigned)])/(nAssigned**2)
    return a-b+c

def kmeans(k, X, kernel):
    assert(k >= 1)
    indices = np.random.randint(k, size=X.shape[0])
    kernelMatrix = kernel(X)

    while True:
        assigned = indices == 0
        distances = np.array([getDistances(kernelMatrix, j, assigned) for j in
                              range(X.shape[0])])
        nextIndices = np.zeros(X.shape[0])
        for i in range(1, k):
            assigned = indices == i
            tempDistances = np.array([getDistances(kernelMatrix, j, assigned)
                                      for j in range(X.shape[0])])
            shorter = tempDistances < distances
            distances[shorter] = tempDistances[shorter]
            nextIndices[shorter] = i

        if (indices == nextIndices).all():
            for i in range(k):
                assigned = indices == i
                plt.plot(X[assigned, 0], X[assigned, 1], 'o',
                         label='Cluster #{}'.format(i+1))
            plt.legend()
            return
        indices = nextIndices

mat = scipy.io.loadmat('hw5_p1b.mat')
X = mat['X']

kmeans(2, X, lambda x : rbfkernel(x, 0.2))
print("end")
##
