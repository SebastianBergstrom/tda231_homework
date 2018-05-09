import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = np.loadtxt('d1.txt')
X = data[:, :2]
Y = data[:, -1]

color = np.where(Y > 0, 'r', 'b')
plt.scatter(data[:, 0], data[:, 1], c=color)

model = SVC(C = 0, kernel='linear')
model.fit(X, Y)
x_vec = np.linspace(-4, 4, 100)
y_vec = np.linspace(-4, 4, 100)
xx, yy = np.meshgrid(x_vec, y_vec)
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

prediction = model.predict(X)
diff = Y-prediction
error_idx = []
for i, val in enumerate(diff):
    if val != 0:
        error_idx.append(i)

plt.scatter(X[error_idx,0],X[error_idx,1], s=200, facecolors='none', edgecolors='black')
plt.contour(xx, yy, Z, 0)
plt.show()
# prediction = model.predict([[-2, -1], [3, -2]])
# print(prediction)
