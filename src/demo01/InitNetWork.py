"""
-*- coding: utf-8 -*-
@file : InitNetWork.py
@description：一个简单的三层神经网络
@author : 杨睿
@time : 2022-01-09 22:20
"""


import numpy as np
import ActivationFunction as Activate
import OutputFunction as Output


def init_network():
    network = {}

    ''' 第一层输入的权重与偏置 '''
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    ''' 第二层输入的权重与偏置 '''
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    ''' 第三层输入的权重与偏置 '''
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    """
    表示 从输入到输出方向的传递处理。与之对应的，是 backward，表示 从输出到输入方向的逆序处理。
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3'],

    """ 计算第一层的输出信号 """
    a1 = np.dot(x, W1) + b1
    z1 = Activate.sigmoid_function(a1)

    """ 计算第二层的输出信号 """
    a2 = np.dot(z1, W2) + b2
    z2 = Activate.sigmoid_function(a2)

    """ 计算第三层的输出信号 """
    a3 = np.dot(z2, W3) + b3

    y = Activate.identity_function(a3)    # 最终输出

    return Output.identity_function(y)


if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)




