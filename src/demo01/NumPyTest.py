"""
-*- coding: utf-8 -*-
@file : NumPyTest.py
@description：NumPy 数组
@author : 杨睿
@time : 2022-01-06 19:33
"""


import numpy as np


if __name__ == '__main__':
    x = np.array([1.0, 2.0, 3.0, 4.0])    # NumPy数组：使用 np.array()方法，接收列表作为对象，生成 NumPy数组
    y = np.array([2.0, 4.0, 6.0, 8.0])
    print(x)
    type(x)

    '''
    NumPy数组的算术运算，需要注意的是，进行两个数组的算术运算时，需确保两个数组的长度一致。
    '''
    print(x+y)    # 输出结果：[ 3.  6.  9. 12.]
    print(x-y)    # 输出结果：[-1. -2. -3. -4.]
    print(x*y)    # 输出结果：[ 2.  8. 18. 32.]
    print(x/y)    # 输出结果：[0.5 0.5 0.5 0.5]

    '''
    NumPy数组的广播，分两种情况：
    （1）数组与单个标量进行运算，此时将数组中的每个元素与标量进行运算，展示最终的结果
    （2）两个不同纬度的数组进行运算，此时将低维数组扩充为高维数组，再进行计算
    '''
    x = np.array([1.0, 2.0, 3.0, 4.0])
    print(x/2.0)    # 2.0：标量。输出结果：[0.5 1. 1.5 2.]

    A = np.array([[1, 2], [3, 4]])
    B = np.array([10, 20])
    print(A*B)    # 输出结果：[ [10 40] [30 80] ]。将数组 B 扩充为 二维数组。

    '''
    NumPy的 N 维数组。注意，多维数组之间的算术运算与一维数组是一样的，二维数组之间的乘法运算并不是数学中矩阵的运算逻辑。 
    '''
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[3, 0], [0, 6]])
    print(A*B)    # 输出结果：[ [3 0] [0 24] ]
    print(A+B)    # 输出结果：[ [4 2] [3 10] ]。该运算逻辑 与 矩阵加法的运算逻辑 一致。对位相加

    '''
    NumPy 数组访问元素
    '''
    X = np.array([[51, 55], [14, 19], [0, 4]])
    print(X[0])        # 输出第 0 行元素：[51, 55]
    print(X[0][1])     # 输出(0,1)的元素：55

    for row in X:
        print(row)     # 遍历输出每行的元素
    
    X = X.flatten()    # 将 X 转换为 一维数组
    print(X)           # 输出结果：[51 55 14 19 0 4]
    print(X[np.array([0, 2, 4])])    # 输出 X 中，索引为 0、2、4 的元素：[51 14  0]
    print(X > 15)        # 比较数组中各个元素是否大于15，得到一个布尔型数组。输出结果：[True True False True False False]
    print(X[X > 15])     # 输出数组中，大于15 的元素。输出结果：[51 55 19]

    '''
    NumPy 的 矩阵乘法
    '''
if __name__ == '__main__':
    A = np.array([1, 2, 3, 4])
    print(np.ndim(A))    # np.ndim(A)：输出 A 的维度
    print(A.shape)       # A.shape：指明 A 是几行几列，以元组的形式输出。在此的输出结果为 (4,)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(np.ndim(B))    # 输出结果：2
    print(B.shape)       # 输出结果：(3, 2)

    C = np.array([[1, 2], [3, 4]])

    print(np.dot(B, C))    # 利用 np.dot(B, C) 实现矩阵的乘法

