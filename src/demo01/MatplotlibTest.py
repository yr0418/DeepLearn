"""
-*- coding: utf-8 -*-
@file : MatplotlibTest.py
@description：图形绘制
@author : 杨睿
@time : 2022-01-06 19:33
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math


if __name__ == '__main__':
    x = np.arange(0, math.pi, 0.1)    # 生成从 0 到 π 的数据，步长为 0.1。即：0，0.1,0.2...
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 绘制图形
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")    # 使用虚线绘制
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin & cos")
    plt.legend()  # 显示两个图形分别用什么线表示
    plt.show()    # 显示图形

    img = imread('../dataSet/testImg/T1.jpg')    # 读取图片
    plt.imshow(img)
    plt.show()
