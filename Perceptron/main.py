from data import generateDataset
import matplotlib.pyplot as plt
from model import Perceptron
import numpy as np

def train(model,x,y,lr,accuracy=None):
    flag = True
    epoch = 0
    while flag:
        epoch += 1
        count = 0
        for i in range(x.shape[0]):
            y_hat = model.forward(x[i, :].reshape(-1, 1))
            if y[i] != y_hat:
                model.update(y_hat, y[i], lr, x[i, :].reshape(-1, 1))
                count += 1
        acc = 1 - count / x.shape[0]
        print("Epoch {}: Accuracy: {:.4f}".format(epoch, acc))
        # 完全分类正确或分类的准确率达到设定的标准
        if not count or ( accuracy and acc >= accuracy):
            flag = False
    
if __name__ == "__main__":
    loc = 5
    scale = 2
    num_samples = 1000
    num_features = 2
    lr = 0.01
    x,y = generateDataset(num_samples, loc, scale, num_features)
    # 生成数据的可视化
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.savefig("org_dataset.png")
    plt.show()
    # 添加阈值对应的固定输入列
    neg_ones = -np.ones((num_samples, 1))
    x = np.hstack((x,neg_ones))
    # 初始化模型
    model = Perceptron(in_feats=num_features)
    # 训练
    train(model, x, y, lr)
    w1,w2,theta = model.w.flatten()
    x1 = np.linspace(-10, 10, 1000)
    y1 = (-w1 * x1 + theta) / w2
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.plot(x1,y1,c='r')
    plt.savefig("outcome.png")
    plt.show()
