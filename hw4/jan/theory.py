##
%matplotlib tk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


y = lambda x : -x + 3
x = np.linspace(0,4)
mSize = 12
classA = np.matrix('2,2;4,4;4,0')
classB = np.matrix('0,0;2,0;0,2')
plt.style.use('fivethirtyeight')
plt.plot(classA[:,0], classA[:,1], 'o', label='Class +1', markersize=mSize)
plt.plot(classB[:,0], classB[:,1], 'o', label='Class -1', markersize=mSize)
plt.plot(x, y(x))
plt.legend()
plt.xticks(np.arange(0, 5, step=1))
plt.yticks(np.arange(-1, 5, step=1))
plt.title('Training points')
plt.savefig('cool.png')
##
