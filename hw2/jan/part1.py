import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
import numpy as np
from scipy.stats import multivariate_normal

def sph_pdf(x, mu, sigmaSquared):
    k = len(mu)
    return np.exp(-1/(2*sigmaSquared)*np.dot(x-mu, x-mu))/np.sqrt((2*np.pi*sigmaSquared)**k)

def sge(X):
    # Precondition
    # X : matrix
    N, p = X.shape
    mu = np.sum(X,axis=0)/N
    sigma = 1/np.sqrt(N*p) * np.sqrt(np.sum( np.linalg.norm(X-mu,axis=1)**2))
    return mu, sigma    

def sph_bayes(Xtest, mus, sigmas):
    # Precondition
    # mus : list of size 2
    # sigmas : list of size 2
    partials = [sph_pdf(Xtest, mu, sigma) for mu, sigma in zip(mus, sigmas)]
    partials = partials/np.sum(partials)
    P1, P2 = partials
    Ytest = 1 if P1 > P2 else -1
    return [P1, P2, Ytest]

def new_classifier(Xtest, mu1, mu2):
    diff = mu1-mu2
    b = 1/2*(mu1 + mu2)
    Ytest = np.sign(np.dot(diff, Xtest-b)/np.linalg.norm(diff))
    return [Ytest]

def new_classifier_wrapper(Xtest, mus, sigmas):
    return new_classifier(Xtest, mus[0], mus[1])

def cross_validation(data, nFolds, classifier):
    # Precondition
    # Data evenly divided by nFolds
    N, p = data.shape
    validation_size = int(N/nFolds)
    validation_indices = np.zeros(N, dtype=bool)
    validation_indices[:validation_size] = 1

    total_error = 0
    for _ in range(nFolds):
        training_data = data[~validation_indices, :]
        validation_data = data[validation_indices, :]
        labels = training_data[:, -1]
        mu1, sigma1 = sge(training_data[labels == 1, :-1])
        mu2, sigma2 = sge(training_data[labels == -1, :-1])
        mus = [mu1, mu2]
        sigmas = [sigma1, sigma2]
        error = 0
        for i in range(validation_size):
            result = classifier(validation_data[i, :-1], mus, sigmas)
            error += (result[-1] - validation_data[i, -1])**2
        error /= validation_size
        validation_indices = np.roll(validation_indices, validation_size)
        total_error += error
    total_error /= nFolds
    return total_error


data = genfromtxt('dataset2.txt', delimiter=',')
labels = data[:, -1]
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
data1 = data[labels == 1, :]
data2 = data[labels == -1, :]
ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], '.', label='Classification 1')
ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], '.', label='Classification 2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Points')
print('abc')
print(cross_validation(data, 5, sph_bayes))
print(cross_validation(data, 5, new_classifier_wrapper))
plt.show()
print('hello')