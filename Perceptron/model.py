import numpy as np

class Perceptron():
    def __init__(self,in_feats) -> None:
        # + 1 means \theta
        self.w = np.random.randn(1, in_feats + 1) * 0.01
    
    def sgn(self,x):
        """
        阶跃函数
        """
        return int(x >= 0)


    def forward(self,x):
        """
        x: 样本 (num_features, 1)
        """
        y_hat = np.dot(self.w,x)
        return self.sgn(y_hat)
    
    def update(self,y_hat,y,lr,x):
        """
        权重调整
        y_hat: 预测值
        y: 真实标签
        lr: 学习率
        x: 样本 (num_features, 1)
        """
        self.w += lr * (y - y_hat) * x.T


        
