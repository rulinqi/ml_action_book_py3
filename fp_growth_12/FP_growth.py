# -*- encoding: utf-8 -*-
"""
    12. FP-growth, 频繁项集
    参考：
"""

class treeNode:
    """
        FP树中节点的定义
    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        """
            将树以文本方式显示
        :param ind:
        :return:
        """
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1):
    """
        FP树构建函数
    :param dataSet:
    :param minSup:
    :return:
    """
    headerTable = {} # 头指针表===>指向给定类型的第一个实例
    # 遍历扫描数据集，统计每个元素出现的频度
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
   # 移除不满足最小支持度的元素项
    # RuntimeError: dictionary changed size during iteration
    # 因为headerTable.keys() 的类型为dict_keys，改为list(headerTable.keys())
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    # headerTable = {k: v for k, v in headerTable.iteritems() if v >= minSup }
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return  None, None

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set',1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] #
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    """
        FP树生长函数
    :param items:
    :param inTree:
    :param headerTable:
    :param count:
    :return:
    """
    # 判断事务中的第一个元素项
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count) # 作为子节点存在,更新元素项的计数
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # 创建子节点
        if headerTable[items[0]][1] == None:  # 更新头指针表  判断元素项是否存在
            headerTable[items[0]][1] = inTree.children[items[0]] #直接添加该元素项在fp树中的指针
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) # 更新头指针表中的链接
    # 对剩余的元素项迭代调用updateTree() ===》 事务中所有元素项，添加到FP树中
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    """
        更新头指针表，确保节点链接指向树中该元素项的每一个实例
    :param nodeToTest:
    :param targetNode:
    :return:
    """
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode # 该元素项添加到链表的末尾

def ascendTree(leafNode, prefixpath):
    """
        迭代上溯整棵树
    :param leafNode:
    :param prefixpath:
    :return:
    """
    if leafNode.parent != None:
        prefixpath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixpath)

def findPrefixPath(basePat, treeNode):
    """
        指定元素生成一个条件模式基
    :param basePat:
    :param treeNode:
    :return:
    """
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
        从条件模式基递归查找频繁项集
    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)



def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    """
        列表转换为字典
    :param dataSet:
    :return:
    """

    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def test_rootNode():
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.disp()

    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()

def test_FPtree():
    simpDat = loadSimpDat()
    print(simpDat)

    initSet = createInitSet(simpDat)
    print(initSet)

    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()

    # test 条件模式基
    # 由于每次传入的dataset字典中的元素顺序不同，构造的fp树不同，产生的条件模式基不同
    print(findPrefixPath('x', myHeaderTab['x'][1]))
    print(findPrefixPath('z', myHeaderTab['z'][1]))
    print(findPrefixPath('r', myHeaderTab['r'][1]))

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)


def test_kosarak():
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)

    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)

    print(myFreqList)



def run_main():
    # test_rootNode()
    # test_FPtree()
    test_kosarak()
    print()

if __name__ == '__main__':
    run_main()