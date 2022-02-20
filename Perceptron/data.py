import numpy as np
import matplotlib.pyplot as plt

def generateDataset(num_samples, loc = 4, scale = 2, num_features = 2):
    """
    生成二分类数据集
    num_samples: 样本数
    num_features: 样本的特征数
    """
    size = num_samples // 2
    # 生成样本
    x1 = np.random.normal(loc, scale, (size, num_features))
    x2 = np.random.normal(-loc, scale, (num_samples - size, num_features))
    x = np.vstack((x1, x2))
    # 生成标签
    y = np.zeros(num_samples)
    y[:size] = 1
    # 打乱数据集
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return x,y

def visualization(x,y,filename):
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    x,y = generateDataset(100)
    visualization(x,y,"org_dataset.png")
