import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].astype(int)
y = np.reshape(y, (y.shape[0], 1))  # 把y转换成mx1的列向量

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plotData(X, y):
    index0 = list()
    index1 = list()
    j = 0
    for i in y:
        if i == 0:
            index0.append(j)
        else:
            index1.append(j)
        j = j + 1
    plt.scatter(X[index0, 0], X[index0, 1], marker='o')
    plt.scatter(X[index1, 0], X[index1, 1], marker='+')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'], loc='upper right')

plotData(X, y)
plt.show()

(m, n) = X.shape
X = np.column_stack((np.ones((m, 1)), X))
(m, n) = X.shape
initial_theta = np.zeros((n, 1))  # nitialize fitting parameters

def costFunction(initial_theta, X, y):
    m = y.shape[0]
    J = np.sum(np.dot((-1 * y).T, np.log(sigmoid(np.dot(X, initial_theta))))- np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(X, initial_theta))))) / m
    return J

def gradient(initial_theta, X, y):
    m, n = np.shape(X)
    initial_theta = initial_theta.reshape((n, 1))
    grad = np.dot(X.T, sigmoid(np.dot(X, initial_theta)) - y) / m
    return grad.flatten()

cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)

Result = op.minimize(fun=costFunction, x0=initial_theta,args=(X, y), method='TNC', jac=gradient)
theta = Result.x
cost = Result.fun
print('Cost at theta found by fminunc:', cost)

def plotDecisionBoundary(theta, X, y):
    figure = plotData(X[:, 1:], y)
    m, n = X.shape
    if n <= 3:
        point1 = np.min(X[:, 1])
        point2 = np.max(X[:, 1])
        point = np.array([point1, point2])
        plot_y = -1 * (theta[0] + theta[1] * point) / theta[2]
        plt.plot(point, plot_y, '-')
        plt.legend(['Admitted', 'Not admitted', 'Boundary'], loc='lower left')
    plt.show()
    return 0

plotDecisionBoundary(theta, X, y)

def predict(theta, X):
    m, n = X.shape
    p = np.zeros((m, 1))
    k = np.where(sigmoid(X.dot(theta)) >= 0.5)
    p[k] = 1
    return p

sample = np.array([1, 45, 85])
prob = sigmoid(np.dot(sample, theta))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)

p = predict(theta, X)
accuracy = np.mean(np.double(p == y)) * 100
print('Train Accuracy:', accuracy)
