import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def SVM(file,kernel,degree=2):
    data = np.loadtxt(file)
    X = data[:, :2]
    Y = data[:, -1]
    soft_margin = 100

    plt.style.use('fivethirtyeight')

    model = SVC(C=soft_margin, kernel=kernel,degree=degree)
    model.fit(X, Y)
    x_vec = np.linspace(-4, 4, 100)
    y_vec = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(x_vec, y_vec)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    prediction = model.predict(X)
    error = Y != prediction

    s_vec = model.support_vectors_
    #plt.scatter(s_vec[:, 0], s_vec[:, 1], s=200, marker='o', color='g')
    plt.scatter(X[error, 0], X[error, 1], s=500, facecolors='none', edgecolors='black', linewidths=3)
    plt.contour(xx, yy, Z, 0, linewidths=1)
    color = np.where(Y > 0, 'r', 'b')
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.title(kernel)
    plt.show()

    #bias = -model.intercept_/model.coef_[0][1]
    #print(bias)

    #support_distances = model.support_vectors_ @ model.coef_[0]/np.linalg.norm(model.coef_[0])
    #print(support_distances)

SVM('d1.txt', 'linear')
SVM('d2.txt', 'linear')
SVM('d2.txt', 'poly')
SVM('d2.txt', 'rbf')
