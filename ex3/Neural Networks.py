import scipy.io as scio
import numpy as np

data = scio.loadmat('ex3data1.mat')
data1 = data.get('X')
label = data.get('y')

def sigmoid(x):
    return 1/(1+np.exp(-x))

parameters = scio.loadmat('ex3weights.mat')

theta1 = parameters.get('Theta1')
theta2 = parameters.get('Theta2')

data_Neur = np.insert(data1, 0, 1, axis=1)
hidden_layer = sigmoid(np.dot(data_Neur, theta1.T))
z_2 = np.insert(hidden_layer, 0, 1, axis=1)
output_layer = sigmoid(np.dot(z_2, theta2.T))
max_prob = np.argmax(output_layer, axis=1)
out = max_prob + 1

accuracy = np.mean(out == label.T)
print('accuracy = {0}%'.format(accuracy * 100))