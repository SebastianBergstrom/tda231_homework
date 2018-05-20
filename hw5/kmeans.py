import numpy as np
import matplotlib.pyplot as plt
import scipy.io
mat = scipy.io.loadmat('hw5_p1a.mat')
print (mat.keys())
X = mat['X']
print(X.shape)

def kmeans(data, k):
    xmin = min(data[:, 0])
    xmax = max(data[:, 0])
    ymin = min(data[:, 1])
    ymax = max(data[:, 1])
    mu = []
    two_iterations_matrix = []
    classification_matrix = np.zeros((data.shape[0], k), dtype=np.int8)
    classification_matrix_old = np.ones((data.shape[0], k), dtype=np.int8)
    mu = data[np.random.choice(data.shape[0], k, replace=False), :]
    #for _ in range(k):
    #   mu.append([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
    it = 0
    print(mu)
    while (classification_matrix_old-classification_matrix).any():
        classification_matrix_old[:] = classification_matrix[:]
        classification_matrix = np.zeros((data.shape[0], k), dtype=bool)
        for index, points in enumerate(data):
            distance = []
            for classification in range(k):
                distance.append(np.linalg.norm(points-mu[classification])**2)
            classified_point = distance.index(min(distance))
            classification_matrix[index, classified_point] = True
        for i in range(k):
            points_in_class = data[classification_matrix[:, i]]
            mu[i] = np.sum(points_in_class, axis=0)/np.sum(classification_matrix[:, i])
        it += 1
        if it == 2:
            two_iterations_matrix = classification_matrix
    print(it)
    return two_iterations_matrix, classification_matrix

A,B = kmeans(X,2)
#print(A)
print('=====')
#print(B)
plt.scatter(X[B[:, 0],0], X[B[:, 0], 1])
plt.scatter(X[B[:, 1], 0], X[B[:, 1], 1])
plt.show()

