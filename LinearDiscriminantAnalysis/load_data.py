from os import startfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_iris(file_path='./data/iris.csv',test_size=0.3):
    """
    功能：加载数据集并划分训练集和测试集
    """
    df = pd.read_csv(file_path)
    x,y = df.iloc[:,0:-1].values,df.iloc[:,-1].values
    #打乱并划分测试集和训练集
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=test_size,random_state=5)
    return train_x,test_x,train_y,test_y
