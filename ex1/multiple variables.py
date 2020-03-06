import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'ex1data2.txt' #导入数据集
data = pd.read_csv(path,names = ['sizes','rooms','prices'])

data['sizes'] = (data['sizes'] - data['sizes'].mean())/data['sizes'].std() #特征缩放
data['rooms'] = (data['rooms'] - data['rooms'].mean())/data['rooms'].std()
data['prices'] = (data['prices'] - data['prices'].mean())/data['prices'].std()

theta = np.matrix(np.zeros(3))
#插入一列，作为x0,值为1
data.insert(0,'ones',1)
cols = data.shape[1]
X = np.matrix(data.iloc[:,0:cols-1])
Y = np.matrix(data.iloc[:,cols-1:cols])
#代价函数
def computeCost(X,Y,theta):
    m = X.shape[0]
    return np.sum(np.power((X*theta.T) - Y,2))/(2*m)
#梯度下降
def gradientDescent(X,Y,theta,alpha,epoch):
    m = X.shape[0]
    cost = []
    for i in range(epoch):
        theta = theta - alpha/m*(X*theta.T - Y).T*X
        cost.append(computeCost(X,Y,theta))       #列表添加元素
    return theta,cost

alpha = 0.01
epoch = 1000
final_theta,cost = gradientDescent(X,Y,theta,alpha,epoch)
final_cost = computeCost(X,Y,final_theta)
print(final_theta)

#画图查看训练过程
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#正规方程法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T*X)*X.T*y
    return theta.T
print(final_theta)
