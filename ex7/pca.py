import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from IPython.display import Image

data = loadmat('ex7data1.mat')
X = data['X']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    return U, S, V
U, S, V = pca(X)
print(U)

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)
Z = project_data(X, U, 1)
print(Z)

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)
X_recovered = recover_data(Z, U, 1)
print(X_recovered )

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()


faces = loadmat('ex7faces.mat')
X = faces['X']

def plot_n_image(X, n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))
    first_n_images = X[:n, :]
    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,sharey=True, sharex=True, figsize=(8, 8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
plot_n_image(X, n=64)
plt.show()

Z = project_data(X, U, k=100)
plot_n_image(Z, n=64)
plt.show()

X_recover = recover_data(Z, U,100)
plot_n_image(X_recover, n=64)
plt.show()