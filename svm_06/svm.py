# -*- encoding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 随机选择j (!=i)
def selectJrand(i,m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
        简化版的SMO求SVM
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C==松弛变量
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return:
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1))) # 拉格朗日乘子 ==对应每个训练样本
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0 # alphas中是否有值进行更新

        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b # 预测结果
            Ei = fXi - float(labelMat[i]) # 误差
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): # alpha能够进行优化
                j = selectJrand(i,m) # 随机选取第二个alpha
                fXj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue

                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T\
                      - dataMatrix[j,:]*dataMatrix[j,:].T  # alphas[j]的最优修改量
                if eta >= 0:
                    print('eta>=0')
                    continue # 退出本次循环 === 进行了简化处理

                # alphas[j]和alphas[i]进行改变，方向相反,大小相同
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L) # 将alphas[j]裁剪在0-C之间
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough');
                    continue
                alphas[i] += labelMat[j] * labelMat[i]*(alphaJold - alphas[j])

                b1 = b - Ei -labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej -labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else: b = (b1 + b2)/2.0

                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0): # 在alphas不进行更新的情况下，迭代maxIter次
            iter += 1
        else: iter = 0
        print('iteration number: %d' % iter)
    return b,alphas

# 使用数据结构来保存所有的数据,方便数据的传递
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # 误差缓存, 矩阵, 第一列==是否有效的标志,第二列==实际的E值

# 计算Ek值
def calcEk(oS, k):
    fxk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
    Ek = fxk -float(oS.labelMat[k])
    return Ek

# 选择第二个alpha, 使步长（Ei-Ej）最大的alpha
def selectJ(i, oS, Ei):
    maxK = -1; maxDelatE = 0;Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] # 返回非零E值所对应的alpha值 # matrix.A 矩阵转化为array数组类型
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDelatE):
                maxk = k; maxDelatE = deltaE; Ej = Ek
        return maxk, Ej
    else: # 第一次循环的话, 随机选择alpha
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 计算误差值并存到缓存中
def updateEk(oS,k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS,Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.alphas[i])
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphaIold)*\
                         oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
                        (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)* \
                         oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
                         (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas): oS.b = b2
        else: oS.b = (b1+b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter: %d i:%d, pairs changed %d' % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print('non-bound, iter: %d i:%d, pairs changed %d' % (iter,i,alphaPairsChanged))
            iter +=1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas


def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1*kTup[1]**2))
    else: raise NameError('Houston we have a Problem --That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # 误差缓存, 矩阵, 第一列==是否有效的标志,第二列==实际的E值
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,i], kTup)

def run_main():
    dataArr, labelArr = loadDataSet('testSet.txt')
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(alphas[alphas>0])
    # print(b)
    #
    # for i in range(100):
    #     if alphas[i] > 0:
    #         print(dataArr[i],labelArr[i])

    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print()

if __name__ == '__main__':
    run_main()