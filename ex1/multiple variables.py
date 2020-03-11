import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

data = loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :-1]
#最后一列为label
y = data[:, -1:]

def feature_normalization(X):
    X_norm = X
    column_mean = np.mean(X_norm, axis=0)
    column_std = np.std(X_norm, axis=0)
    X_norm = X_norm - column_mean
    X_norm = X_norm / column_std
    return column_mean, column_std, X_norm
means, stds, X_norm = feature_normalization(X)

m = len(y)
X_norm = np.hstack((np.ones((m,1)), X_norm))
theta = np.zeros((X_norm.shape[1], 1))

def computeCost(X, y, theta):
    h = np.dot(X, theta)
    J = 1/(2*m) * sum((h-y)**2)
    return J[0]
computeCost(X_norm, y, theta)

alpha = 0.01;
iterations = 400;
#梯度下降
def gradientDescent(X, y, theta, alpha, iterations):
     J_history = np.zeros((iterations, 1))
     for i in range(iterations):
        h = np.dot(X, theta)
        k = np.dot(np.transpose(X) ,(h-y))
        theta = theta - alpha * k / m
        J_history[i] = computeCost(X, y, theta)
     return theta, J_history
theta, J_history= gradientDescent(X_norm, y, theta, alpha, iterations)
print(theta)

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta
theta2=normalEqn(X_norm, y)
print(theta2)

def plotJ(J_history, iterations):
    x = np.arange(1, iterations+1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations vs loss')
    plt.show()
plotJ(J_history, iterations)

def test(means, stds, theta):
    t1 = np.array([[1650,3]])
    t1 = t1 - means
    t1 = t1 / stds
    t1 = np.hstack((np.ones((t1.shape[0], 1)), t1))
    res = np.dot(t1, theta)
    print('predict house price:', res[0][0])
test(means, stds, theta2)