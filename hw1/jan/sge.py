import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def sge(X):
    n, p = X.shape
    mu = np.sum(X, 0)/n
    diff = X - mu
    sigma = np.sqrt(np.sum(diff**2)/(n*p))
    return mu, sigma

def myplot1():
    plt.style.use('ggplot')
    data = np.loadtxt("dataset0.txt")
    data = data[:,:2]
    mu, sigma = sge(data)
    plt.plot(data[:,0], data[:,1], '.', label='Data points', markersize=4)
    x = np.linspace(150, 850)
    y = np.linspace(70, 780)
    X, Y = np.meshgrid(x, y)
    n, _ = data.shape
    levels = [(k*sigma)**2 for k in range(1,4)]
    distances = np.sum((data-mu)**2, 1)
    labels = ["{:.2f}% of all points outside of curve".format(np.count_nonzero(level < distances)/n*100) for level in levels]
    CS = plt.contour(X, Y, (X-mu[0])**2 + (Y-mu[1])**2, levels, colors=[cm.Set1(i) for i in range(1,5)])
    for label, collection in zip(labels, CS.collections):
        collection.set_label(label)
    plt.legend()
    plt.axis('equal')

myplot1()
plt.show()
