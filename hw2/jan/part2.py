import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets

digits = datasets.load_digits()

data = digits.data
print(data.shape)
target_names = digits.target_names
print (target_names)
y = digits.target
plt.matshow(digits.images[399])
plt.show()

def sge(X):
    # Precondition
    # X : matrix
    N, p = X.shape
    mu = np.sum(X,axis=0)/N
    sigma = 1/np.sqrt(N*p) * np.sqrt(np.sum( np.linalg.norm(X-mu,axis=1)**2))
    return mu, sigma

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
        labels = digits.target
        mu1, sigma1 = sge(training_data[labels == 5])
        mu2, sigma2 = sge(training_data[labels == 8])
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

data_5 = np.hstack((np.array(data[digits.target == 5]), np.array([1]*len(data[digits.target == 5]))))

digits_58 =  np.concatenate((data[digits.target == 5],data[digits.target == 8]),axis=0)

digits_class = digits.target
