# -*- encoding:utf-8 -*—

from bayes_04.bayes import *
from numpy import *
import feedparser
import operator

def calcMostFreq(vocabList, fullText):
    freqDict = {} # 词频统计
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedFreq)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList = []; classList = []; fullText = [];
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList) # append方法可以接收任意数据类型的参数，并且简单地追加到list尾部
        fullText.extend(wordList) # extend方法只能接收list，且把这个list中的每个元素添加到原list中
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    # 移除文本统计中top30高频词==或使用 停用词表 去除
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(2*minLen)); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNBO(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList, p0v, p1v


def getTopWords(ny,sf):
    vocabList, p0V, p1V = localWords(ny,sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -5.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -5.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    print(sortedSF)
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    print(sortedNY)

def run_main():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, pSF, pNY = localWords(ny,sf)
    getTopWords(ny, sf)

    # print(ny['entries'])
    # print(len(ny['entries']))
    print()

if __name__ == '__main__':
    run_main()
