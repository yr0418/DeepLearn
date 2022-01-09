"""
-*- coding: utf-8 -*-
@file : OutputFunction.py
@description：输出函数
@author : 杨睿
@time : 2022-01-09 23:29
"""


import numpy as np


def identity_function(y):
    """
    恒等函数，即本身
    """
    return y


def softmax_function(x):
    """
    softmax 函数
    """
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_a = np.sum(exp_x)
    y = exp_x / sum_exp_a
    return y
