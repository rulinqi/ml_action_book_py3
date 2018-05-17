# -*- encoding: utf-8 -*-
"""
    10. K-means
    参考：

"""

from numpy import *
import matplotlib.pyplot as plt

# set_printoptions(True)

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 计算两向量的欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB,2)))

# 构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) # 每个点的簇分配结果矩阵(簇索引值，误差值)
    centroids = createCent(dataSet, k) # 构建k个随机质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: # 簇分配情况发生改变
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k): # 重新计算质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分k-均值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 初始簇
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1]) # 该簇划分之后的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 其他簇的SSE
            print('sseSplit, and notSplit:',sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 改变本次划分簇的分配簇名
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is:',bestCentToSplit)
        print('the len of bestClustAss is:',len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 更新重新划分的簇的质心
        centList.append(bestNewCents[1,:].tolist()[0]) # 添加新的质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss # 更新新的簇划分结果

    return mat(centList), clusterAssment


# 画聚类图
def plt_clusters(datMat,myCentroids,clustAssing):
    m = shape(myCentroids)[0]
    clust_marker = ['s','o','^','8','p','d','v','h','>','<']
    for i in range(m):
        cluster = datMat[nonzero(clustAssing[:, 0] == i)[0], :]
        plt.scatter(cluster[:, 0].T.A[0], cluster[:, 1].T.A[0], s=30, c='k', marker=clust_marker[i])
    plt.scatter(myCentroids[:, 0].T.A[0], myCentroids[:, 1].T.A[0], s=100, c='r', marker='+')
    plt.show()

# 计算地球表面两点间的距离
def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0,0]-vecA[0,0]) / 180)
    return arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split()
        datList.append([float(lineArr[-1]), float(lineArr[-2])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0',**axprops)
    imgP = plt.imread('Portland.png')  # 基于图像创建矩阵
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],ptsInCurrCluster[:,1].flatten().A[0],marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],marker='+',s=300)
    plt.show()


def run_main():
    # datMat = mat(loadDataSet('testSet.txt'))
    # print(min(datMat[:,0]),min(datMat[:,1]),max(datMat[:,0]),max(datMat[:,1]))
    # print(randCent(datMat, 2))
    # print(distEclud(datMat[0],datMat[1]))
    #
    # myCentroids, clustAssing = kMeans(datMat,4)
    # print(myCentroids)
    # # print(clustAssing)
    # plt_clusters(datMat,myCentroids, clustAssing)
    #
    #
    # datMat3 = mat(loadDataSet('testSet2.txt'))
    # centList,myNewAssments = biKmeans(datMat3,3)
    # print(centList)
    #
    # plt_clusters(datMat3,centList,myNewAssments)

    clusterClubs(5)
    print()

if __name__ == '__main__':
    run_main()