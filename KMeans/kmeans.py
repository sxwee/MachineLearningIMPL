import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class KMeansModel:
    def rand_pick(self, x, k):
        """
        随机选取k个簇中心
        """
        n = x.shape[0]
        indices = np.random.choice(n, k, replace=False)
        return x[indices]

    def calculate_distance(self, x, centers):
        """
        计算簇中心与数据样本之间的欧式距离
        centers: 簇中心数据 (k, d)
        x: 样本 (N, d)
        """
        return cdist(x, centers, metric="cosine")

    def get_centers(self, k, x, y):
        """
        根据计算结果重新计算簇中心
        y: 根据距离将数据集划分的标签数组 (N)
        """
        centers = np.zeros((k, x.shape[1]))
        for label in range(k):
            centers[label] = np.mean(x[y == label], axis=0)
        return centers

    def get_label(self, dis):
        """
        根据距离矩阵将每个样本划分到距离最小的簇中心
        """
        return np.argmin(dis, axis=-1)

    def cluster(self, x, k, times):
        """
        进行KMeans聚类
        x: 数据样本 (N, d)
        k: 类别数
        tims: 迭代次数
        """
        # 随机选取k个作为初始簇中心
        centers = self.rand_pick(x, k)
        for _ in range(times):
            # 计算各个样本到簇中心的距离
            dis = self.calculate_distance(x, centers)
            # 根据距离矩阵将样本进行划分
            y = self.get_label(dis)
            # 重新计算新的簇中心
            centers = self.get_centers(k, x, y)
        return y


if __name__ == "__main__":
    k, num_iters = 3, 500
    n_components = 2
    iris_data = pd.read_csv("../datasets/iris.csv").values
    x, y = iris_data[:, :-1], iris_data[:, -1]
    model = KMeansModel()
    y_hat = model.cluster(x, k, num_iters)
    # sklearn库
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x)
    y_sk = kmeans.labels_
    # pca降维
    pca = PCA(n_components)
    pca_x = pca.fit_transform(x)
    # 可视化
    plt.figure(figsize=(9,6))
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y_hat)
    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y)
    plt.subplot(1, 3, 3)
    plt.title("Prediction(sklearn)")
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y_sk)
    plt.savefig("kmeans_vis.png")
    plt.show()
