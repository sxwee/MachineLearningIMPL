import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataset1(size=1000):
    """
    功能：生成一元线性模型数据集
    size：数据集大小
    """
    x = np.random.randn(size,1)
    y = x + np.random.randn(size,1) + 1.2
    
    return x,y

def loadDataset2(size=1000):
    """
    功能：生成多元线性模型数据集
    size：数据集大小
    """
    x1 = np.random.randn(size,1)
    x2 = np.random.randn(size,1)
    x3 = np.random.randn(size,1)
    y = 0.8*x1 + 1.7*x2 + 6.6*x3 + np.random.randn(size,1) + 5.2
    ones = np.ones(shape=(size,1))
    #构造(w,b)形式
    x = np.hstack((x1,x2,x3,ones))
    
    return x,y



if __name__ == "__main__":
    size = 1000
    x1,y1 = loadDataset1(size=size)
    #一元线性模型参数求解方法
    w1 = np.dot(y1.T,x1-np.mean(x1)) / (np.sum(x1**2) - 1.0 /size * np.sum(x1)**2)
    b1 = 1.0 / size * np.sum((y1-w1*x1))
    print(w1[0,0],b1)
    
    #一元线性模型划为矩阵形式后进行求解
    ones = np.ones(shape=(size,1))
    #构造(w,b)形式
    x1 = np.hstack((x1,ones))
    print(np.linalg.inv(x1.T.dot(x1)).dot(x1.T).dot(y1))
    
    #多元线性模型超级参数求解
    x2,y2 = loadDataset2(size=size)
    print(np.linalg.inv(x2.T.dot(x2)).dot(x2.T).dot(y2))
