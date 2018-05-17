# -*- encoding: utf-8 -*-
"""
    2. k-近邻算法
    http://blog.csdn.net/niuwei22007/article/details/49703719

"""

import numpy as np
import operator
import os
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')     # 设置图片显示的主题样式

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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

def autoNorm(dataSet):
    """
    归一化数值
    :param dataSet:
    :return:
    """
    # 参数0==函数从列中选取最小值 ,无参数==从矩阵中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet / np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    分类器测试
    :return:
    """
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(os.path.join(dataset_path, filepath))
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        # 前10%为测试集，后90%为训练集,近邻数为3
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with %d, the real answer is %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
    """
    约会网站应用分类器测试
    :return:
    """
    resultList = ['not at all','in small doses','in large doses']

    # 1、在python2.x中raw_input( )和input( )，两个函数都存在，其中区别为
    # raw_input( )---将所有输入作为字符串看待，返回字符串类型
    # input( )-----只能接收“数字”的输入，在对待纯数字输入时具有自己的特性，它返回所输入的数字的类型（ int, float ）
    # 2、在python3.x中raw_input( )和input( )进行了整合，去除了raw_input( )，仅保留了input( )函数，其接收任意任性输入，
    # 将所有输入默认为字符串处理，并返回字符串类型。
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent fliter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix(os.path.join(dataset_path, filepath))
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult - 1])

def img2vector(filename):
    """
    32*32的图像矩阵转换为1*1024的向量
    :param filename:
    :return:
    """
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('./dataset/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('./dataset/digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('./dataset/digits/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./dataset/digits/trainingDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

def run_main():
    # 1. 测试knn
    #test_knn()

    # # 2. exp 改进约会网站的配对效果
    # datingDataMat, datingLabels = exp_dating_knn()
    #
    # # 3. 归一化数值
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    #
    # print(normMat)
    # print(ranges)
    # print(minVals)

    # 分类器测试
    # datingClassTest()

    # 约会网站应用分类器测试
    # classifyPerson()

    # 图像矩阵转换测试
    # testVector = img2vector('./dataset/digits/trainingDigits/0_13.txt')
    # print(testVector.shape)
    # print(testVector[0,0:31])

    # 手写数字识别系统测试
    handwritingClassTest()


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
    #  http://www.cnblogs.com/shanlizi/p/6850318.html --散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('玩游戏-消耗冰淇淋')
    plt.xlabel('玩视频游戏所耗时间百分比')
    plt.ylabel('每周消费的冰淇淋公升数')
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    # 15 * 。。增加标记点的粗细
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15 * np.array(datingLabels), 15 * np.array(datingLabels))
    plt.show()

    # 3. 双散点图
    # 将画布分割成1行2列，使用1行
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('玩游戏-消耗冰淇淋')
    plt.xlabel('玩视频游戏所耗时间百分比')
    plt.ylabel('每周消费的冰淇淋公升数')
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    # 15 * 。。增加标记点的粗细
    ax1.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15 * np.array(datingLabels), 15 * np.array(datingLabels))

    ax2 = fig.add_subplot(122)
    ax2.set_title('飞行里程-玩游戏')
    plt.xlabel('每年获取的飞行常客里程数')
    plt.ylabel('玩视频游戏所耗时间百分比')
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    # 15 * 。。增加标记点的粗细
    ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15 * np.array(datingLabels), 15 * np.array(datingLabels))

    plt.show()

    return datingDataMat, datingLabels


if __name__ == '__main__':
    run_main()


