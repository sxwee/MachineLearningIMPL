import numpy as np
from numpy import linalg
from data import generateDataset,spiltDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def initializeBeta(feature_nums):
    """
    功能：初始化参数Beta=(w;b)
    """
    beta = np.zeros(shape=(feature_nums,1))
    beta[-1] = 1
    return beta

def train(train_x,train_y,iterations=100,learning_rate=0.001):
    """
    功能：模型训练
    train_x,train_y：训练数据特征集，训练数据标签集
    iterations：迭代次数
    """
    train_x,train_y = train_x.T,train_y.reshape(-1)
    beta = initializeBeta(train_x.shape[0])
    for i in range(iterations):
        beta_x = np.dot(beta.T[0], train_x)
        lbeta = np.sum(-1 * train_y*beta_x + np.log(1 + np.exp(beta_x)))
        dbeta = np.dot(train_x,-1*train_y + np.divide(np.exp(beta_x),1+np.exp(beta_x))).reshape(-1,1)
        beta -= dbeta*learning_rate

        print('iteraing {} times lbeta is {}'.format(i + 1,lbeta))

    return beta

def test(test_x,test_y,beta):
    """
    功能：计算测似集的准确率
    test_x,test_y：测试数据特征集，测试数据标签集
    """
    #计算出预测值
    pred_y = np.dot(beta.T[0],test_x)
    #预测值y>0则为正例1，否则为反例0
    pred_y[pred_y >= 0] = 1
    pred_y[pred_y < 0] = 0
    #计算预测的准确率
    bingo = 0
    for i,yi in enumerate(pred_y):
        if yi == test_y[i]:#预测正确
            bingo += 1
    #print(bingo,test_y.shape[0])
    return bingo / test_y.shape[0]

if __name__ == "__main__":
    points,x,y = generateDataset()
    train_x,test_x,train_y,test_y = spiltDataset(x,y)
    beta = train(train_x,train_y)
    print(beta)
    test_x,test_y = test_x.T,test_y.reshape(-1)
    acc = test(test_x,test_y,beta)
    print('the acc of test dataset is {}'.format(acc))
    x11,x12,x21,x22 = points
    plt.scatter(x11,x12,c='b',marker='o')
    plt.scatter(x21,x22,c='orange',marker='v')
    x3 = np.random.randint(0,15,size=1000)
    w1,w2,b = beta[0][0],beta[1][0],beta[2][0]
    y3 = -1*(w1*x3 + b) / w2
    plt.plot(x3,y3,c='r')
    plt.show()


