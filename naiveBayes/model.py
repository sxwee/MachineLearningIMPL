import numpy as np

class naiveBayes():
    def __init__(self,):
        """
        cols: 标识样本属性是离散的还是连续的
            1: 离散属性
            0: 连续
        example: cols = [1,1,0,1]
        """
        pass

    def fit(self,x,y):
        """
        x: ndarray, (n_samples, n_features)
        y: ndarray, (n_samples, )
        """
        self.params = self.getMeanAndStd(x,y)

    def predict(self,x):
        """
        x: ndarray, (n_samples, n_features)
        """
        flag,probs = False,None
        for pc,mu_c,std_c in self.params:
            pxc = self.normal_distribution(x,mu_c,std_c)
            if not flag:
                flag = True
                probs = (pxc.prod(axis=1) * pc).reshape(-1,1)
            else:
                pxc = (pxc.prod(axis=1) * pc).reshape(-1,1)
                probs = np.hstack([probs, pxc])
        return np.argmax(probs, axis=1)



    def getMeanAndStd(self,x,y):
        """
        计算连续属性的均值和标准差,即计算正态分布的参数
        """
        params = []
        for c in np.unique(y):
            tx = x[y==c]
            params.append([np.sum(y == c) / y.shape[0], tx.mean(axis=0),tx.std(axis=0)])
        return params
    
    def normal_distribution(self,x,mu,sigma):
        """
        功能: 计算P(x|c)
        x: 样本属性,(n_samples, n_features)
        mu: 均值列表,(N,)
        sigma: 标准差,(N, )
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1 * np.power(x - mu, 2) / (2 * np.power(sigma,2)))

if __name__ == "__main__":
    pass