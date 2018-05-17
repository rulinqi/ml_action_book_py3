# -*- encoding: utf-8 -*-
"""
    11. Apriori，频繁项集，关联分析
    参考：http://blog.csdn.net/sinat_17196995/article/details/71124284

"""

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    """
        构建第一个候选项集的列表C1
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # map()函数：接收函数 f 和 list，并通过把f() 依次作用在 list 的每个元素上，得到一个新的 list 并返回
    # py2中map()函数直接分为list对象,python3中map()函数直接返回的是map对象需要加list()
    return list(map(frozenset,C1)) # frozenset类型: 不可变集合,可以作为字典键值使用

def scanD(D, Ck, minSupport):
    """
        计算所有频繁项集
    :param D: 数据集
    :param Ck: 候选项集列表
    :param minSupport: 最小支持度
    :return: 满足最低要求的项集构成集合 retList ,候选项中各个数据的支持度 supportData
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid): # 判断候选项中是否含数据集的各项 ==要求can的类型为frozenset类型--可直接作为字典键值
                # if not ssCnt.has_key(can):  # python3 can not support
                if not can in ssCnt:
                    ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = [] #L1初始化
    supportData = {} #记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport :
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk,k):
    """
        由Lk==>Ck
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return:  retList 候选项集列表==Ck
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i]) [:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j]) # A | B ： python中集合的并操作
    return retList

def apriori(dataSet, minSupport = 0.5):
    """
        apriori算法的主函数
    :param dataSet:
    :param minSupport:
    :return:
    """
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def rulesFromConseq(freqSet, H, supportData, brl, minConf):
    """
        生成候选规则集合
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmpl = aprioriGen(H, m+1)
        Hmpl = calcConf(freqSet, Hmpl, supportData, brl,minConf)
        if (len(Hmpl) > 1):
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
        评估关联规则
    :param freqSet: 频繁项集
    :param H: 可以出现在规则右部的元素列表
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'---->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def generateRules(L, supportData, minConf = 0.7):
    """
        生成关联规则，主函数
    :param L: 频繁项集列表
    :param supportData: 包含频繁项集列表支持数据的字典
    :param minConf: 最小可信度阀值
    :return: bigRuleList 包含可信度的规则列表
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def L_test():
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print(C1)
    D = list(map(set, dataSet))
    print(D)
    L1, suppData0 = scanD(D, C1, 0.5)
    print(L1)

def apriori_test():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    print(L)
    print(L[0])

    aprioriGen(L[0],2)

    L, suppData = apriori(dataSet, minSupport=0.7)
    print(L)

def rules_test():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet, minSupport=0.5)
    rules = generateRules(L, suppData, minConf=0.7)
    print(rules)

    # rules = generateRules(L, suppData, minConf=0.5)
    # print(rules)

def run_main():
    # L_test()
    # apriori_test()
    rules_test()
    print()

if __name__ == '__main__':
    run_main()