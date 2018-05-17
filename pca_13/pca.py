# -*- encoding: utf-8 -*-
"""
    13. PCA
    参考：

"""

from numpy import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

set_printoptions(suppress=True) # numpy不使用科学计数法输出

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr] #
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    """
        主成分分析-PCA

    :param dataMat: 操作的数据集
    :param topNfeat: 应用到N个特征上，可选
    :return:
    """
    meanVals = mean(dataMat, axis=0) # 求均值
    meanRemoved = dataMat - meanVals # 归一化数据
    covMat = cov(meanRemoved, rowvar=0) # 求协方差
    eigVals,eigVects = linalg.eig(mat(covMat)) # 求特征值和特征向量
    eigValInd = argsort(eigVals) # 对特征值进行排序，默认从小到大
    eigValInd = eigValInd[:-(topNfeat+1):-1] # 逆序取得特征值最大的元素
    redEigVects = eigVects[:,eigValInd] # 用特征向量构成矩阵
    lowDDataMat = meanRemoved * redEigVects  # 用归一化后的各个数据与特征矩阵相乘，映射到新的空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  #还原原始数据
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) # 计算非NaN值的平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat


def run_main():
    # dataMat = loadDataSet('testSet.txt')
    # lowDMat, reconMat = pca_13(dataMat,2)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    # ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='r')
    # plt.show()
    # print()

    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)  # 求均值
    meanRemoved = dataMat - meanVals  # 归一化数据
    covMat = cov(meanRemoved, rowvar=0)  # 求协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 求特征值和特征向量
    print(eigVals)
    eigVals_sort = sort(eigVals)
    eigVals_sort = eigVals_sort[:-21:-1]
    full0 = eigVals[eigVals==0].size/eigVals.size
    insum = eigVals_sort/sum(eigVals)
    print('特征值为0的占比:%.2f%%' % (full0*100))
    print('主成分方差百分比: ')
    print(insum)
    [round(x,2) for x in insum]

    plt.plot(range(1,21),insum,'-o')
    plt.show()
    print()

if __name__ == '__main__':
    run_main()