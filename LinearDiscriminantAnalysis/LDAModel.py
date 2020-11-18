import numpy as np

class LDA(object):
    """
    线性判别分析
    """
    def __init__(self,data,target,d) -> None:
        self.data = data
        self.target = target
        self.d = d
        self.labels = set(target)
        self.mu = self.data.mean(axis=0)
    
    def divide(self):
        """
        功能：将传入的数据集按target分成不同的类别集合并求出对应集合的均值向量
        """
        self.classify,self.classmu = {},{}
        for label in self.labels:
            self.classify[label] = self.data[self.target==label]
            self.classmu[label] = self.classify[label].mean(axis=0)
    
    def getSt(self):
        """
        功能：计算全局散度矩阵
        """
        self.St = np.dot((self.data-self.mu).T,(self.data-self.mu))

    def getSb(self):
        """
        功能：计算类内散度矩阵
        """
        self.Sb = np.zeros((self.data.shape[1],self.data.shape[1]))
        for i in self.labels:
            #获取类别i样例的集合
            classi = self.classify[i]
            #获取类别i的均值向量
            mui = self.classmu[i]
            self.Sb += len(classi) * np.dot((mui - self.mu).reshape(-1,1),(mui - self.mu).reshape(1,-1))

    def getW(self):
        """
        功能：计算w
        """
        self.divide()
        self.getSt()
        self.getSb()
        #St = Sw + Sb
        self.Sw = self.St - self.Sb 
        #计算Sw-1*Sb的特征值和特征向量
        #eig_vectors[:i]与 eig_values相对应
        eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(self.Sw).dot(self.Sb))
        #寻找d个最大非零广义特征值
        topd = (np.argsort(eig_values)[::-1])[:self.d]
        #用d个最大非零广义特征值组成的向量组成w
        self.w = eig_vectors[:,topd]
        



if __name__ == "__main__":
    x = np.array([[1,2,3],[2,1,3],[2,4,1],[1,3,2],[3,6,4],[3,1,1]])
    y = np.array([0,1,2,0,1,2])
    lda = LDA(x,y,2)
    lda.getW()