import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets

digits = datasets.load_digits()

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
    for it in range(nFolds):
        if it == nFolds-1:
            validation_indices[:-N%nFolds] = 0
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

def extractDigit (d,c):
    data = digits.data
    digits_d = data[digits.target == d]
    n,_ = digits_d.shape
    classification = np.zeros((n,1))+c
    return np.hstack((digits_d,classification))

digits_58 = np.concatenate((extractDigit(5,1),extractDigit(8,-1)),axis=0)
np.random.shuffle(digits_58)
print(cross_validation(digits_58,5,new_classifier_wrapper))

rescaled_digits = digits_58[:,:-1]/16
length = rescaled_digits.shape[0]
digits_var = np.zeros((length,16))

for i in range(length):
    mat = np.reshape(rescaled_digits[i,:],(8,8))
    var_rows = mat.var(axis=1)
    var_cols = mat.var(axis=0)
    digits_var[i,:] = np.concatenate((var_rows.T,var_cols))

digits_var_58 = np.hstack((digits_var,digits_58[:,-1].reshape(length,1)))
np.random.shuffle(digits_var_58)
print(cross_validation(digits_var_58,5,new_classifier_wrapper))



