import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt('ex2data1.txt', delimiter=',')

m = data.shape[0]
X = np.array(data[:,[0,1]]).reshape(m,2)
Y = np.array(data[:,2]).reshape(m,1)

#normalize
# for i in range(0, X.shape[1] - 1):
#     X[:,i] = X[:,i]/(X[:,i].max() - X[:,i].min())

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


theta = np.zeros((1, 3))
cost = np.array(costFn(theta=theta, x=X, y=Y)).reshape(1, 1)

# the following from @kaleko github
def optimizeTheta(theta, x, y):
    result = optimize.fmin(costFn,
                           x0=theta,
                           args=(x, y),
                           maxiter=400,
                           full_output=True)
    return result[0], result[1]

optimalTheta = optimizeTheta(theta, x=X, y=Y)
# Out[13]: (array([-25.16130062,   0.20623142,   0.20147143]), 0.2034977015902151)
# if normalizing the X in the begining, the optimization doesn't work

# splitting the data into two classes, and plot them separately
# more convenient than using the color argument
pos = X[np.where(Y == 1)[0], :]
neg = X[np.where(Y == 0)[0], :]
plt.plot(pos[:, 1], pos[:, 2], 'o', c='green')
plt.plot(neg[:, 1], neg[:, 2], 'o', c='red')
x1_values = np.array([X[np.argmin(X[:, 1]), 1], X[np.argmax(X[:, 1]), 1]])
x2_values = - optimalTheta[0][0] / optimalTheta[0][2] - optimalTheta[0][1] * x1_values / optimalTheta[0][2]
plt.plot(x1_values, x2_values)
