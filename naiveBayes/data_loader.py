import pandas as pd
import numpy as np

def standardization(data):
    """
    z-score归一化,即（X-Mean）/(Standard deviation)
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def loadWine(data_path='../datasets/wine.csv',split=0.8):
    """
    加载葡萄酒数据集的数据和标签
    data_path: 数据集的路径
    split: 训练集/测试集划分比, 默认为8:2
    """
    wine = pd.read_csv(data_path).values
    # 打乱数据集
    indices = np.arange(wine.shape[0])
    np.random.shuffle(indices)
    wine = wine[indices]
    # 划分训练集和测试集
    x,y = standardization(wine[:,1:]), wine[:,[0]].flatten().astype(np.int)
    # 类标由1,2,3变成0,1,2
    y -= 1
    train_size = int(x.shape[0] * split)
    train_x,train_y,test_x,test_y = x[:train_size,:],y[:train_size],\
                                        x[train_size:,:],y[train_size:]
    return train_x,train_y,test_x,test_y

if __name__ == "__main__":
    rain_x,train_y,test_x,test_y = loadWine()