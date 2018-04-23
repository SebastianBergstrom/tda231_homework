

import os
from numpy import genfromtxt

data = genfromtxt('dataset2.txt', delimiter=',')
labels: object = data[:,-1]



