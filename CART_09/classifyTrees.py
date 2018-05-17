# -*- encoding: utf-8 -*-
"""
    CART,gini系数
    参考：https://www.jianshu.com/p/01d820be67fb
          http://www.cnblogs.com/pinard/p/6053344.html(原理)
    CART回归树和CART分类树的建立和预测的区别主要有下面两点：
    1)连续值的处理方法不同
    2)决策树建立后做预测的方式不同。
"""

import numpy as np

def Impurity(X,Label,ClassNum = 3):
    Len = np.size(X)
    sX = np.sort(X)
    Tha = np.zeros(Len-1)
    gi = np.zeros(Len-1)
    for i in range(Len-1):
        Th = (sX[i] + sX[i+1])/2
        Tha[i] = Th
        idx1 = np.where(X < Th)
        idx2 = np.where(X >= Th)
        p = np.zeros([2,ClassNum])
        g = np.zeros([1,2])
        ww = np.zeros([2,1])
        for Ti in range(2):
            if Ti == 1:
                idxTP = idx1
            else:
                idxTP = idx2
            Lab = Label[idxTP]
            for cs in np.arange(1,ClassNum+1):
                if np.size(idxTP) == 0:
                    p[Ti,cs-1] = 0
                else:
                    p[Ti,cs-1] = np.size(np.where(Lab == cs)) / np.size(idxTP)
            g[0,Ti] = gini(p[Ti,:])
            ww[Ti,0] = np.size(idxTP) / Len
            gi[i] = np.dot(g,ww)
            del idxTP,Lab
    idxa = np.argmin(gi)
    ThA = Tha[idxa]
    impur = gi[idxa]
    return impur,ThA

def gini(p):
# 基尼系数计算公式
    if np.all(p == 0):
        g = 0
    else:
        g = 1 - np.sum(np.square(p))
    return g

X = np.array([0,1,5,4,3,4,5,6,8,7,9,0])
Label = np.array([1,1,1,1,2,2,2,2,3,3,3,3])
impur,ThA = Impurity(X,Label)
print("impurity=",impur,"Best dividing point:",ThA)
