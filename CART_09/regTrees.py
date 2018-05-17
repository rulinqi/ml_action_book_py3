# -*- encoding: utf-8 -*-
"""
    09. 树回归
    参考：http://blog.csdn.net/sinat_17196995/article/details/69621687

"""

import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t') # 移除空格，以制表符分割
        fltLine = list(map(float, curLine)) # py3    str-->float
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
        将数据集切分为两个子集
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    # np.nonzero返回非零元素的索引
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0],:]  # 书上错误
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]  # 书上错误
    return mat0, mat1

# Tree结点类型：回归树
def regLeaf(dataSet):
    # 生成叶结点，在回归树中是目标变量特征的均值
    return np.mean(dataSet[:,-1])

# 误差计算函数：回归误差
def regErr(dataSet):
    # 计算目标的平方误差（均方误差*总样本数）
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] # 允许的误差下降值
    tolN = ops[1] # 切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: # 倒数第一列转化成list 不重复
        return None, leafType(dataSet) # 如果剩余特征数为1，停止切分
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0; # np.inf==无穷大
    for featIndex in range(n-1):
        # 遍历每个特征里不同的特征值
        for splitVal in set((dataSet[:, featIndex].T.tolist())[0]): # py3 书上错 set(dataSet[:, featIndex])
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) # 对每个特征进行二元分类
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: # 更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS: # 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] <tolN) or (np.shape(mat1)[0] < tolN): #判断切分后子集大小，小于最小允许样本数停止切分
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return  retTree

# 判断是否是课树
def isTree(obj):
    return (type(obj).__name__=='dict')

# 返回树的平均值==从上往下遍历到叶节点为止
def getMean(tree):
    if isTree(tree['right']) :
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']) :
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0

def prune(tree, testData):
    """
        树的后剪枝
    :param tree: 待剪枝的树
    :param testData:剪枝所需的测试数据
    :return:
    """
    if np.shape(testData)[0] == 0: # 确认数据集非空
        return getMean(tree)
    # 假设发生过拟合，采用测试数据对树进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):  # 左右子树非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) # 测试数据集进行切分,lSet, rSet也为测试数据
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], rSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): # 剪枝后判断是否还是有子树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left']+tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else: return tree
    else: return tree

#将数据集格式化为X Y
def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:, 0:n-1]; Y = dataSet[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0: # X Y用于简单线性回归，需要判断矩阵可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
                    try increasing the second value of ops')
    ws = xTx.I * (X.T * Y) # 简单线性回归，求斜率ws
    return ws, X, Y

#不需要切分时生成模型树叶节点
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

#用来计算误差找到最佳切分
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

def run_main():
    test_bike()
    print()

def test_reg():
    # testMat = np.mat(np.eye(4))
    # print(testMat)
    #
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print('mat0')
    # print(mat0)
    # print('mat1')
    # print(mat1)
    # print()

    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    # print(createTree(myMat))
    # print(createTree(myMat, ops=(10000,4)))
    # print(createTree(myMat, ops=(0,1)))

    myDat2 = loadDataSet('exp2.txt')
    myMat2 = np.mat(myDat2)

    # myTree = createTree(myMat2, ops=(0,1))
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDatTest)
    # print(prune(myTree, myMat2Test))

    print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))

def test_bike():
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    print('回归树：')
    myTree = createTree(trainMat, ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    print(np.corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])

    print('模型树:')
    myTree = createTree(trainMat,modelLeaf,modelErr, (1,20))
    yHat = createForeCast(myTree, testMat[:,0], modelTreeEval)
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    print('简单线性模型：')
    ws, X, Y = linearSolve(trainMat)
    print(ws)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

if __name__ == '__main__':
    run_main()