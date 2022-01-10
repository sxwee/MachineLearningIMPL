from model import naiveBayes
from data_loader import loadWine
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

def main():
    # 调用自己生成的模型
    nb_m = naiveBayes()
    train_x,train_y,test_x,test_y = loadWine()
    nb_m.fit(train_x,train_y)
    pred_y = nb_m.predict(test_x)
    # sklearn中的朴素贝叶斯模型
    nb_sk = GaussianNB()
    nb_sk.fit(train_x,train_y)
    pred_y1 = nb_sk.predict(test_x)
    print("Sklearn Model Acc: {:.4f}, Our Model Acc: {:.4f}".format(
        accuracy_score(pred_y1, test_y),
        accuracy_score(pred_y, test_y)
    ))

    pca_sk = PCA(2)
    lx_sk = pca_sk.fit_transform(test_x)
    print(lx_sk.shape)
    plt.subplot(1,3,1)
    plt.title("Our NaiveBayes")
    plt.scatter(lx_sk[:,0],lx_sk[:,1],c=pred_y)
    plt.subplot(1,3,2)
    plt.title("Sklearn NaiveBayes")
    plt.scatter(lx_sk[:,0],lx_sk[:,1],c=pred_y1)
    plt.subplot(1,3,3)
    plt.title("Original")
    plt.scatter(lx_sk[:,0],lx_sk[:,1],c=test_y)
    plt.savefig("vis.png")
    plt.show()

if __name__ == "__main__":
    main()