import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data1.txt', delimiter=',')

m = data.shape[0]
X = np.array(data[:,[0,1]]).reshape(m,2)
Y = np.array(data[:,2]).reshape(m,1)

#normalize
for i in range(0, X.shape[1] - 1):
    X[:,i] = X[:,i]/(X[:,i].max() - X[:,i].min())

X = np.c_[np.ones((m,1)),X]

def logisticLink(x):
    return (1.0 + np.exp(-x))**(-1.0)

def costFn(theta, x, y):
    # this is the h_\theta(x) in lecture slides 6
    # in matrix form for all rows of X
    hypothesis = logisticLink(np.dot(X, np.transpose(theta)))

    cost = (np.dot(np.transpose(y), np.log(hypothesis)) + np.dot(np.transpose(1.0 - y), np.log(1.0-hypothesis))) / (-x.shape[0])
    return cost

def gradientDescent(theta, x, y):
    hypothesis = logisticLink(np.dot(X, np.transpose(theta)))
    gd = np.dot(np.transpose(hypothesis - y), x) / float(x.shape[0])
    return gd


alpha = 0.05
theta = np.zeros((1, 3))
cost = np.array(costFn(theta=theta, x=X, y=Y)).reshape(1, 1)
