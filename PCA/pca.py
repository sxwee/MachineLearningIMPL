import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

class PCAModel():
    def __init__(self,n_components) -> None:
        """
        n_components: 降维后的维度/重构阈值
        """
        self.n_components = n_components

    def transform(self,data):
        """
        data: (nums,features)
        """
        # 中心化
        data = data - np.mean(data,axis=0)
        # 协方差矩阵
        data = np.matmul(data.T,data)
        # v[:,i]是对应特征值w[i]的特征向量
        e_vals,e_vecs = np.linalg.eig(data)
        # 特征值从大到小排序
        indices = np.argsort(-e_vals)
        # 根据重构阈值获取d'
        if self.n_components >= 0 and self.n_components <= 1.0:
            k,val_cur,val_sum = 0,0,np.sum(e_vals)
            while True:
                val_cur += e_vals[indices[k]]
                k += 1
                if val_cur / val_sum >= self.n_components:
                    break
            self.n_components = k
        # w (features, d')
        w = e_vecs[:,indices[:self.n_components]]
        return w
        

if __name__ == "__main__":
    # 重构阈值
    n_components = 0.95
    data = pd.read_csv("../datasets/iris.csv").values
    x,y = data[:,:4],data[:,-1]
    # 自己实现的模型
    pca = PCAModel(n_components)
    w = pca.transform(x)
    lx = np.matmul(x,w)
    # sklearn自带的模型
    pca_sk = PCA(n_components)
    lx_sk = pca_sk.fit_transform(x)
    plt.subplot(1,2,1)
    plt.title("Our PCA")
    plt.scatter(lx[:,0],lx[:,1],c=y)
    plt.subplot(1,2,2)
    plt.title("Sklearn")
    plt.scatter(lx_sk[:,0],lx_sk[:,1],c=y)
    plt.savefig("vis.png")
    plt.show()