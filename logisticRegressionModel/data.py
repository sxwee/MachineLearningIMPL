import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generateDataset(train_rate=0.7,size=1000):
    """
    功能：生成数据集并按7:3划分训练集和测试集
    train_rate：训练集所占比例，剩下的即为测试集
    size：数据集大小
    """
    #两种标签的数据量各占一半
    label_size = size // 2

    #标签为0的数据点生成
    x11 = np.random.normal(4,1.8,label_size).reshape(-1,1)
    x12 = np.random.normal(4,1.8,label_size).reshape(-1,1)
    y1 = np.zeros(shape=(label_size,1))

    #标签为1的数据集点生成
    x21 = np.random.normal(12,2.3,label_size).reshape(-1,1)
    x22 = np.random.normal(12,2.3,label_size).reshape(-1,1)
    y2 = np.ones(shape=(label_size,1))

    #合并成完整的数据集
    x1 = np.vstack((x11,x21))
    x2 = np.vstack((x12,x22))
    ones = np.ones(shape=(size,1))
    x = np.hstack((x1,x2,ones))
    y = np.vstack((y1,y2))

    return (x11,x12,x21,x22),x,y

def spiltDataset(datas,labels,test_size=0.3):
    """
    功能：划分训练集，测试集，验证集
    datas,labels：数据,标签
    train_size,test_size,valie_size：各集合所占的比例
    """
    train_x,test_x,train_y,test_y = train_test_split(datas,labels,test_size=test_size,random_state=5)

    return train_x,test_x,train_y,test_y

if __name__ == "__main__":
    _,x,y = generateDataset()
    train_x,test_x,train_y,test_y = spiltDataset(x,y)
    print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)