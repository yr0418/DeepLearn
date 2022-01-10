"""
-*- coding: utf-8 -*-
@file : showImg.py
@description：mnist 手写数字识别
@author : 杨睿
@time : 2022-01-10 20:35
"""

import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]    # 选取训练集中的第一张图像
    label = t_train[0]
    print(label)

    print(img.shape)             # (784, )
    img = img.reshape(28, 28)    # 将图像的形状变成原来的尺寸
    print(img.shape)             # (28, 28)
    
    img_show(img)    # 输出图像
