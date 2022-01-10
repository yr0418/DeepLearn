# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    """
    获取实验数据
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
    获取权重和偏置函数
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    三层神经网络对输入的图像进行预测
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def predict_single():
    """
    每次只输入一张图像，输出一个预测结果
    """
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0

    # 每次处理一张图像
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 预测的正确度


def predict_batch():
    """
    批处理，每次处理一批数据
    """
    x, t = get_data()
    network = init_network()

    batch_size = 100  # 批数量
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i+batch_size]    # 列表的截取操作，一张图片为一行，共 100 行
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)    # axis=1，即行方向，取每行的最大值所在的索引。若axis=0，即列方向。
        accuracy_cnt += np.sum(p == t[i: i+batch_size])    # 比较 预测结果与实际结果，用 “==” 生成 True/False 构成的布尔型数组，计算其中 True 的数量。

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == '__main__':
    predict_batch()