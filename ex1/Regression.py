import numpy
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

data = pd.read_table(r"ex1data1.txt", header=None, delimiter=',') #装载数据
plt.scatter(data.iloc[0:97, 0], data.iloc[0:97, 1]) #数据散点图

lrModel = LinearRegression() #估计模型参数，建立回归模型
x = data.iloc[0:97, 0].values.reshape(-1, 1)
y = data.iloc[0:97, 1]
lrModel.fit(x,y) #模型训练
lrModel.score(x,y) #对回归模型进行检验
lrModel.predict([[60],[70]]) #利用回归模型进行预测
alpha = lrModel.intercept_ #查看截距
beta = lrModel.coef_ #查看参数
alpha + beta*numpy.array([60,70])
plt.plot(x, lrModel.predict(x), color='red', linewidth=2) #拟合的直线
plt.show()

#代价函数
#p_y是predict y，即模型的输出
def cost(p_y,y):
    return 0.5*numpy.power(y-p_y,2).mean()
def linear(theta0,theta1,x1):
    return theta0+theta1*x1

theta0=numpy.linspace(-8,12,100)
theta1=numpy.linspace(-1,4,100)
theta0,theta1=numpy.meshgrid(theta0,theta1)
cost_=[]
for t0,t1 in zip(theta0.flat,theta1.flat):
    cost_.append(cost(linear(t0,t1,x),y.values.reshape(-1, 1)))
cost_=numpy.array(cost_).reshape((100,100))

fig=plt.figure(figsize=(9,3))
ax=fig.add_subplot(121)
levels=[1,2,4,8,16,32,64,128]
c=ax.contour(theta0,theta1,cost_,levels,colors='k',linewidths=0.5)
ax.clabel(c,fontsize=10)
cf=ax.contourf(theta0,theta1,cost_,levels,cmap='YlOrRd')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax=fig.add_subplot(122,projection='3d')
ax.plot_surface(theta0,theta1,cost_,cmap='coolwarm',alpha=0.8)
ax.contour(theta0,theta1,cost_,levels,colors='k',linewidths=0.5)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('cost')
ax.view_init(45,135)
plt.show()
