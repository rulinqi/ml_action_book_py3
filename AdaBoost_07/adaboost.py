# -*- encoding: utf-8 -*-
"""
    07. 利用AdaBoost元算法提高分类性能
    参考：

"""
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 通过指定特征阀值比较,对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
        通过指定特征阀值比较,对数据进行分类
    :param dataMatrix:
    :param dimen: 分类特征
    :param threshVal: 分类的阀值
    :param threshIneq: 标记：小于lt，大于gt
    :return:
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
        构建单层决策树
    :param dataArr: 分类数据
    :param classLabels: 分类标签
    :param D:  权重向量
    :return: bestStump{dim, thresh, ineq}=最佳单层决策树{特征，分类阀值，大小}, minError=错误率, bestClasEst=类别估计值
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n): # 遍历数据集上的所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:, i].max();
        setpSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1): # 按numSteps遍历次数，遍历特征所有值
            for inequal in ['lt', 'gt']: # 小于和大于不等式间切换
                threshVal = (rangeMin + float(j) * setpSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) # 进行单特征分类预测
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 # 标记分类错误的
                weightedError = D.T * errArr # 计算加权错误率
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' %
                #       (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=4.0):
    """
        基于单层决策树的AdaBoost训练
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return:
    """
    weakClassArr = [] # 弱分类器集合
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m) # 每个数据点的权重向量
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels,D) # 构建单层决策树
        # print('D: ',D.T)
        # 根据本次分类器的错误率计算权重
        alpha = float(0.5*log((1.0-error) / max(error,1e-16))) # max(error,1e-16)==避免除0错误
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst.T)
        # 为下一次的迭代计算D
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha*classEst
        # print('aggClassEst: ',aggClassEst.T)
        # 错误率累计计算
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print('total error: ', errorRate,'\n')
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
    """
        AdaBoost分类器
    :param datToClass: 待分类数据
    :param classifierArr: 弱分类器集合
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)

# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def run_main():
    # D = mat(ones((5,1)) / 5)
    # datMat, classLabels = loadSimpData()
    # # print(buildStump(datMat, classLabels, D))
    #
    # classifierArray = adaBoostTrainDS(datMat, classLabels,30)
    #
    # adaClassify([0,0], classifierArray)
    # print('=======')
    # adaClassify([[5,5],[0,0]], classifierArray)

    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = mat(ones((67,1)))
    err = errArr[prediction10!=mat(testLabelArr).T].sum()
    print(err/67)
    print()

if __name__ == '__main__':
    run_main()
