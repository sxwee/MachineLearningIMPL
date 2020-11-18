from LDAModel import LDA
from load_data import load_iris
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score

def judge(gauss_dist,x):
    """
    功能：判断样本x属于哪个类别
    """
    #将样本带入各个类别的高斯分布概率密度函数进行计算
    outcome = [[k,norm.pdf(x,loc=v['loc'],scale=v['scale'])] for k,v in gauss_dist.items()]
    #寻找计算结果最大的类别
    outcome.sort(key=lambda s:s[1],reverse=True)
    return outcome[0][0]

def Test():
    """
    功能：对测试集进行分类并返回准确率
    """
    #加载数据集
    train_x,test_x,train_y,test_y = load_iris()
    #创建模型
    lda = LDA(train_x,train_y,1)
    #获取投影矩阵w
    lda.getW()
    #对训练集进行降维
    train_x_new = np.dot((train_x), lda.w)
    #获取训练集各个类别对应的高斯分布的均值和方差
    gauss_dist = {}
    for i in lda.labels:
        classi = train_x_new[train_y==i]
        loc = classi.mean()
        scale = classi.std()
        gauss_dist[i] = {'loc':loc,'scale':scale}
    test_x = np.dot(test_x,lda.w)
    pred_y = np.array([judge(gauss_dist,x) for x in test_x])
    
    return accuracy_score(test_y,pred_y)
    

if __name__ == "__main__":
    acc = Test()
    print(acc)


