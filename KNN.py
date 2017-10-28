# -*- encoding: utf-8 -*-
"""
    2. k-近邻算法
    http://blog.csdn.net/niuwei22007/article/details/49703719

"""

import numpy as np
import operator

def createDataSet():
    group = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
        k-近邻算法
    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻的数目
    :return:
    """
    # 1. 计算距离， 欧氏距离
    dataSetSize = dataSet.shape[0]
    # tile（A, reps）返回一个shape=reps的矩阵，矩阵的每个元素是A
    # 比如 A=[0,1,2] 那么，tile(A, 2)= [0, 1, 2, 0, 1, 2]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 2. 选择距离最小的可k个点
    # 按照升序进行快速排序，返回的是原数组的下标。
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedDistIndicies = distances.argsort()

    classCount = {}
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 3. 排序
    # python2 中 classCount.iteritems()
    # http://blog.csdn.net/dongtingzhizi/article/details/12068205
    # reverse 倒序 ,key为函数，指定取待排序元素的哪一项进行排序
    # operator.itemgetter函数用于获取对象的哪些维的数据
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def run_main():
    group, labels = createDataSet()
    print(group)
    print(labels)
    k = classify0([0,0], group, labels, 3)
    print(k)

if __name__ == '__main__':
    run_main()


