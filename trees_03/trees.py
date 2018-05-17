# -*- encoding: utf-8 -*-
"""
    03. 决策树
    参考：

"""

from math import log
import operator
import trees_03.treePlotter as treePlotter

def calcShannonEnt(dataSet):
    """
        计算信息熵   信息熵越高，混合的数据也越多
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {} # 所有可能的分类，统计类别出现次数
    for featVec in dataSet:
        currentLabel = featVec[-1] # 可能分类
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0 # 信息熵 ,计算所有类别 所有可能值 包含的 信息期望值(熵)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
        划分数据集，根据给定的特征，返回划分好的给特征的值的数据
    :param dataSet:  待划分的数据集(list)
    :param axis:  划分数据集的特征
    :param value:  需要返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: # 抽取符合特征值的集合:去除分类特征，将符合的数据添加的list
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # list.extend(b):只添加b中的元素[... ...]
            retDataSet.append(reducedFeatVec) # list.append(b)：将b(list)直接添加到list中 [... [...]]
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
        选取特征，划分数据集，计算出最好的划分数据集的特征
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet) # 初始信息熵
    bestInfoGain = 0.0 ; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 获取该特征的标签列表
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 根据特征给定标签划分数据
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy # 信息增益 = 初始信息熵 - 特征划分后的信息熵
        if (infoGain > bestInfoGain): # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
        多数表决de投票
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True) #使用次数对vote排序
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
        递归创建树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet] # 获取数据集的标签
    if classList.count(classList[0]) == len(classList): # 所有类的标签完全相同,返回该类的标签
        return classList[0]
    if len(dataSet[0]) == 1: # 使用完了所有特征，
        return majorityCnt(classList) # 返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] # 为了不改变原始的labels列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    """
        使用决策树分类
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    """
        决策树的存储
    :param inputTree:
    :param filename:
    :return:
    """
    import pickle #python序列化对象，这里序列化保存树结构的字典对象
    fw = open(filename, 'wb') # 书上错误
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb') # 书上错误
    return pickle.load(fr)

def run_main():
    # myDat, labels = createDataSet()
    # print(myDat)
    # print('calcShannonEnt:')
    # print(calcShannonEnt(myDat))
    #
    # # myDat[0][-1] = 'maybe'
    # # print(myDat)
    # # print('calcShannonEnt:')
    # # print(calcShannonEnt(myDat))
    #
    # print(splitDataSet(myDat,0,1))
    # print(splitDataSet(myDat,0,0))
    # print(splitDataSet(myDat, 1, 1))
    #
    # print(chooseBestFeatureToSplit(myDat))
    #
    # myTree = createTree(myDat, labels)
    # print(myTree)

    # myDat, labels = createDataSet()
    # print(labels)
    # myTree = treePlotter.retrieveTree(0)
    # print(myTree)
    # print(classify(myTree,labels,[1,0]))
    #
    # storeTree(myTree,'classfierStorage.txt')
    # grabtree = grabTree('classfierStorage.txt')
    # print(grabtree)

    lenses_classify()

def lenses_classify():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)

if __name__ == '__main__':
    run_main()