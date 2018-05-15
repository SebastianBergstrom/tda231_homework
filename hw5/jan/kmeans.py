##
%matplotlib tk
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def kmeans(k, X, plot=False):
    assert(k >= 1)

    mu = np.random.rand(k, 2)
    if plot:
        point, = plt.plot(mu[:,0], mu[:,1], 'o')

    while True:
        distances = np.linalg.norm(mu[0,:]-X, axis=1)
        indices = np.zeros(X.shape[0])
        for i in range(1,k):
            tempDistances = np.linalg.norm(mu[i,:]-X, axis=1)
            shorter = tempDistances < distances
            distances[shorter] = tempDistances[shorter]
            indices[shorter] = i


        muTemp = np.zeros((k, 2))
        for i in range(k):
            assigned = indices == i
            muTemp[i,:] = sum(X[assigned])/np.count_nonzero(assigned)

        if (mu == muTemp).all():
            return

        mu = muTemp
        if plot:
            point.set_xdata(mu[:,0])
            point.set_ydata(mu[:,1])
            plt.draw()
            plt.pause(0.15)



mat = scipy.io.loadmat('hw5_p1a.mat')
X = mat['X']
plt.plot(X[:,0], X[:,1] , '.')
kmeans(6, X, True)
##
