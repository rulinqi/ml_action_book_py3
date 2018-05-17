# -*- encoding: utf-8 -*-
"""
    03. 决策树,使用Matplotlib绘图
    参考：

"""

import matplotlib.pylab as plt
import matplotlib as mpl

#
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
    xytext=centerPt,textcoords='axes fraction', va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot_first():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('叶节点', (0.8,0.1), (0.3,0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    """
        获取叶节点的数目
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #py3中
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
        获取树的层数
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  # py3中
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    """
        计算父节点和子节点的中间位置，添加文本
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    """
    xMid = (parentPt[0]-cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    """
        绘制树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff) #
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW  #
            plotNode(secondDict[key], (plotTree.x0ff,plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD #下次重新调用时恢复y

def createPlot(inTree):
    """
        绘制图像的主函数
    :param inTree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree)) # 树宽
    plotTree.totalD = float(getTreeDepth(inTree)) # 树深
    plotTree.x0ff = -0.5/plotTree.totalW; plotTree.y0ff = 1.0; #
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def run_main():
    # createPlot()

    # print(retrieveTree(1))
    myTree = retrieveTree(0)
    createPlot(myTree)
    myTree['no surfacing'][3] = 'maybe'
    print(myTree)
    createPlot(myTree)
    # print(getNumLeafs(myTree))
    # print(getTreeDepth(myTree))
    print()


if __name__ == '__main__':
    run_main()