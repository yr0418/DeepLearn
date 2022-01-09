"""
-*- coding: utf-8 -*-
@file : ActivationFunction.py
@description：激活函数
@author : 杨睿
@time : 2022-01-06 19:33
"""


import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """
    阶跃函数的实现
    函数式:    当 x < 0，输出 0；当 x > 0，输出 1
    :param x: NumPy数组
    :return:  int型的、只含有 0和1 的输出信号数组
    """
    y = x>0
    return y.astype(np.int)    # 将布尔型数组 y 转换为 整型，其中 false 转为 0，true 转为 1

    # 简易写法：return np.array(x>0, dtype=np.int)


def sigmoid_function(x):
    """
    sigmoid 函数的实现
    函数式:    1 / [1+e^(-x)]
    :param x: NumPy数组，在此利用 NumPy 的广播功能，得到的结果也是一个 NumPy数组。
    :return:  根据函数式计算得到的结果组成的 NumPy数组
    """
    return 1/(1+np.exp(-x))    # np.exp(-x) 即为 e^(-x)


def ReLU_function(x):
    """
    ReLU 函数的实现
    函数式:    当 x > 0，输出 x；当 x < 0，输出 0
    :param x: NumPy数组
    :return:  根据函数式计算得到的结果组成的 NumPy数组
    """
    return np.maximum(0, x)    # np.maximum：从输入的数值中，选择最大值进行输出


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1,)
    y1 = step_function(x)
    y2 = sigmoid_function(x)
    y3 = ReLU_function(x)
    plt.plot(x, y1, label="step", linestyle="--")
    plt.plot(x, y2, label="sigmoid")
    plt.plot(x, y3, label="ReUL")
    plt.ylim(-0.1, 1.1)    # 指定 y 轴的范围
    plt.legend()
    plt.show()

