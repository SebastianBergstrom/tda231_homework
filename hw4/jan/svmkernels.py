##
%matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
data = np.loadtxt('d1.txt')
X = data[:,:2]
Y = data[:,-1]
clf = SVC(kernel='linear')
clf.fit(X, Y)

mSize = 12
edgeWidth = 5
padding = 0.5

f = lambda x, w1, w2, b: -(w1*x + b)/w2
plt.style.use('fivethirtyeight')
classA = Y == 1
plt.plot(X[classA,0], X[classA,1], 'o', label='Class +1', markersize=mSize)
plt.plot(X[~classA,0], X[~classA,1], 'o', label='Class -1', markersize=mSize)
predictions = clf.predict(X)
wronglyLabeled = predictions != Y


plt.plot(X[wronglyLabeled, 0], X[wronglyLabeled, 1], 'o',
         markerFaceColor='none', markeredgewidth=edgeWidth,
         markerSize=mSize+edgeWidth, label='Failed to classify correctly')
plt.plot(X[clf.support_, 0], X[clf.support_, 1], 'o',
         markerFaceColor='none', markeredgewidth=edgeWidth,
         markerSize=mSize+edgeWidth, label='Support vector')
x = np.linspace(min(X[:,0])-padding, max(X[:,0])+padding)
plt.plot(x, f(x, *clf.coef_[0], clf.intercept_))
plt.legend()
##
