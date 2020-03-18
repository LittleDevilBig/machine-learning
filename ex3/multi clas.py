import scipy.io as scio
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = scio.loadmat('ex3data1.mat')
data1 = data.get('X')
label = data.get('y')

def plot_100_image(x):
    sample_idx = np.random.choice(np.arange(x.shape[0]), 100)
    sample_images = x[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for row in range(10):
        for col in range(10):
            ax_array[row, col].matshow(sample_images[10 * row + col].reshape((20, 20)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
plot_100_image(data1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunction(theta, x, y, Lambda):
    m = np.shape(x)[0]
    thetaReg = theta[1:]
    y = y.transpose()
    hypothesis = sigmoid(np.dot(x, theta))
    cost = y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)
    reg = np.sum(thetaReg * thetaReg) * Lambda/(2 * m)
    costAll = np.mean(-cost) + reg
    return costAll

def Gradient(theta, x, y, Lambda):
    m = np.shape(x)[0]
    theteReg = theta[1:]
    y = y.transpose()
    hypothesis = sigmoid(np.dot(x, theta))
    loss = hypothesis - y
    cost_1 = np.dot(loss, x)/m
    reg = np.concatenate([np.array([0]), (Lambda / m) * theteReg])
    gradient = cost_1 + reg
    return gradient

def oneVsAll(x, y, Lambda, k):
    all_theta = np.zeros((k, np.shape(x)[1]))
    for i in range(k):
        theta = np.zeros(np.shape(x)[1])
        y_i = np.array([1 if label == i+1 else 0 for label in y])
        ret = minimize(fun=costFunction, x0=theta, args=(x, y_i, Lambda), method='TNC', jac=Gradient, options={'disp': True})
        all_theta[i, :] = ret.x
    return all_theta

def predictOneVsAll(x, all_theta):
    thetaT = np.transpose(all_theta)
    probMat = sigmoid(np.dot(x, thetaT))
    maxProb = np.argmax(probMat, axis=1)
    label = maxProb+1
    return label

dataFix = np.insert(data1, 0, 1, axis=1)
thetaAll = oneVsAll(dataFix, label, 1, 10)
pred = predictOneVsAll(dataFix, thetaAll)
accuracy = np.mean(pred == label.T)
print('accuracy = {0}%'.format(accuracy * 100))

