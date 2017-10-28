# -*- encoding: utf-8 -*-
"""
    2. k-近邻算法
    http://blog.csdn.net/niuwei22007/article/details/49703719

"""

import numpy as np
import operator
import os
import matplotlib.pyplot as plt

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

def file2matrix(filename):
    """
    将文本记录转换为Numpy的解析程序
    :return:
    """
    # 1. 得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 2.创建返回的Numpy矩阵
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    # 3.解析文件数据到列表
    for line in arrayOLines:
        # 把回车符号给去掉
        line = line.strip()
        # 把每一行数据用\t分割
        listFromLine = line.split('\t')
        # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        returnMat[index, :] = listFromLine[0:3]
        # 把该样本对应的标签放至标签集，顺序与样本集对应。
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
dataset_path = './dataset'
filepath = 'datingTestSet2.txt'


def run_main():
    # 1. 测试knn
    #test_knn()

    # 2. exp 改进约会网站的配对效果
    exp_dating_knn()


def test_knn():
    #测试knn
    group, labels = createDataSet()
    print(group)
    print(labels)
    k = classify0([0,0], group, labels, 3)
    print(k)

def exp_dating_knn():
    """
    knn 改进约会网站的配对效果
    :return:
    """
    # 1. 载入数据
    datingDataMat, datingLabels = file2matrix(os.path.join(dataset_path, filepath))
    print(datingLabels[0:20])

    # 2. 使用Matplotlib创建散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15 * np.array(datingLabels),15 * np.array(datingLabels))
    plt.show()



if __name__ == '__main__':
    run_main()


