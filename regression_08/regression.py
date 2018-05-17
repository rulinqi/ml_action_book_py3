# -*- encoding: utf-8 -*-
"""
    0.8 Regression
    参考：https://cloud.tencent.com/developer/article/1052837===

"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0: # 计算一个数组的行列式,判断是否能计算逆矩阵
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权线性回归===给待测点附近的每个点赋予一定的权重
# 程序的计算量增加,预测每个点时使用整个数据集
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat=mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) # 使用高斯核，赋予权重 # 权重大小以指数级衰减,使用k来调节
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar # 数据的标准化处理

    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat): # 数据标准化
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt): # 迭代次数
        print(ws.T)
        lowestError = inf;
        for j in range(n): # 对每个特征
            for sign in [-1,1]: # 增大或缩小
                wsTest = ws.copy()
                wsTest[j] += eps*sign # 依据步长eps增加或减小，获得新的ws
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)  # 计算新ws下的误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

from bs4 import BeautifulSoup

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
        从页面读取数据，生成retX和retY列表
    :param retX:数据x
    :param retY:数据y
    :param inFile:HTML文件
    :param yr:年份
    :param numPce:乐高部件数目
    :param origPrc:原价
    :return:
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)

def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99) # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)

def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList) # 将列表中的元素打乱
        for j in range(m): # 划分数据，训练集和测试集
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY) # 岭回归
        for k in range(30):
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain # 使用训练时的参数将测试数据标准化
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,array(testY))
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX  # 数据还原
    print('the best model from Ridge Regression is:\n',unReg)
    print('with constant term: ',-1*sum(multiply(meanX,unReg)) + mean(yMat))



def run_main():

    # xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[:2])
    # ws = standRegres(xArr,yArr)
    # print(ws)
    #
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat * ws
    #
    # print(corrcoef(yHat.T,yMat)) #预测与真实序列的相关系数
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat=xCopy*ws
    # ax.plot(xCopy[:,1],yHat)
    # plt.show()

    # xArr, yArr = loadDataSet('ex0.txt')
    # print(yArr[0])
    # yhat = lwlr(xArr[0],xArr,yArr,1.0)
    # print(yhat)
    # yhat = lwlr(xArr[0], xArr, yArr, 0.01)
    # print(yhat)
    #
    # yHat = lwlrTest(xArr,xArr,yArr,0.003)
    # xMat = mat(xArr)
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0],s=2,c='red')
    # plt.show()


    # abX,abY = loadDataSet('abalone.txt')
    # yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    # yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    #
    # print(rssError(abY[0:99],yHat01.T))
    # print(rssError(abY[0:99], yHat1.T))
    # print(rssError(abY[0:99], yHat10.T))
    #
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    #
    # print(rssError(abY[100:199], yHat01.T))
    # print(rssError(abY[100:199], yHat1.T))
    # print(rssError(abY[100:199], yHat10.T))
    #
    # ws = standRegres(abX[0:99],abY[0:99])
    # yHat = mat(abX[100:199]) * ws
    # print(rssError(abY[100:199],yHat.T.A))

    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX,abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    # xArr,yArr = loadDataSet('abalone.txt')
    # stageWise(xArr,yArr,0.001,5000)
    # print('--------------------------------')
    #
    # xMat = mat(xArr)
    # yMat = mat(yArr).T
    # xMat = regularize(xMat)
    # yM = mean(yMat,0)
    # yMat = yMat - yM
    # weights = standRegres(xMat,yMat.T)
    # print(weights.T)

    lgX = []; lgY = []
    setDataCollect(lgX,lgY)
    # print(shape(lgX))
    lgX1 = mat(ones((63,5)))
    lgX1[:,1:5] = mat(lgX)
    # print(lgX[0])
    # print(lgX1[0])
    ws = standRegres(lgX1,lgY)
    print(ws)
    # print(lgX1*ws)

    crossValidation(lgX,lgY,10)
    print()

if __name__ == '__main__':
    run_main()