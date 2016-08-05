import numpy as np
import matplotlib.pyplot as plt

###Single X case
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = np.array(data[:,0]).reshape(m,1)
Y = np.array(data[:,1]).reshape(m,1)

plt.plot(X,Y,'rx')
plt.ylabel('Profit in $10k')
plt.xlabel('Population of city in 10k')
plt.show()

X = np.c_[np.ones((m,1)),X]
theta = np.zeros((1, 2))
iterations = 1000
alpha = .01

def costFn(theta):
    if len(np.transpose(theta)) != 2:
        return 'Expect the initial theta length to be 2, one for intercept one for slope'
    else:
        cost = np.mean(np.square(np.dot(X, np.transpose(theta)) - Y))/2
        return cost

cost = np.array(costFn(theta=theta)).reshape(1,1)

i = 0
while i < iterations:
    oldTheta = theta[theta.shape[0] - 1, :].reshape(1, theta.shape[1])
    oldCost = cost[cost.shape[0] - 1, :].reshape(1, cost.shape[1])

    gradientDescent = np.array([np.mean(np.dot(X, np.transpose(oldTheta)) - Y) * alpha
                                   , np.mean(
            np.multiply(np.dot(X, np.transpose(oldTheta)) - Y, X[:, 1].reshape(X.shape[0], 1))) * alpha
                                ]).reshape(1, theta.shape[1])

    newTheta = oldTheta - gradientDescent
    newCost = np.array(costFn(newTheta)).reshape(1, 1)

    theta = np.r_[theta, newTheta]
    cost = np.r_[cost, newCost]
    i = i + 1


###plot the data points and the Gradient descent regression line

x_values = np.array([X[np.argmin(X[:,1]),], X[np.argmax(X[:,1]),]])
y_hat = np.dot(x_values, np.transpose(theta[theta.shape[0] - 1, :]))

plt.plot(X[:,1],Y,'rx')
plt.plot(x_values[:,1],y_hat)
plt.ylabel('Profit in $10k')
plt.xlabel('Population of city in 10k')
plt.show()

#########################################
###         Multi-variate X case      ###
#########################################
data = np.loadtxt('ex1data2.txt', delimiter=',')
m = data.shape[0]
X = np.array(data[:,[0,1]]).reshape(m,2)
#normalize
for i in range(0, X.shape[1]):
    X[:,i] = X[:,i]/(X[:,i].max() - X[:,i].min())
X = np.c_[np.ones((m,1)),X]
Y = np.array(data[:,2]).reshape(m,1)

theta = np.zeros((1, 3))
iterations = 1500
alpha = .01

#the only difference from the single X version is the error message
def costFn(theta):
    if len(np.transpose(theta)) != 3:
        return 'Expect the initial theta length to be 2, one for intercept one for slope'
    else:
        cost = np.mean(np.square(np.dot(X, np.transpose(theta)) - Y))/2
        return cost

cost = np.array(costFn(theta=theta)).reshape(1,1)

i = 0
while i < iterations:
    oldTheta = theta[theta.shape[0] - 1, :].reshape(1, theta.shape[1])
    oldCost = cost[cost.shape[0] - 1, :].reshape(1, cost.shape[1])

    gradientDescent = (np.dot(np.transpose(np.dot(X, np.transpose(oldTheta)) - Y), X))*alpha/m

    newTheta = oldTheta - gradientDescent
    newCost = np.array(costFn(newTheta)).reshape(1, 1)

    theta = np.r_[theta, newTheta]
    cost = np.r_[cost, newCost]
    i = i + 1

plt.plot(cost,'rx')