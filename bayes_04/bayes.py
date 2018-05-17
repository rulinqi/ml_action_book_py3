# -*- encoding:utf-8 -*—

from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 创建字典
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 并集
    return  list(vocabSet)

# 词集模型==只看词是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

# 词袋模型==记录词出现的次数
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 文档数
    numWords = len(trainMatrix[0]) # 文档内单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 类别1所占的比例
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0

    # 对文档分类时，多个概率相乘，避免其中一个概率为0，乘积为0，则初始化为1
    p0Num = ones(numWords);
    p1Num = ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 向量相加==统计对应词出现的个数
            p1Denom += sum(trainMatrix[i]) # 分类中词的总数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    # 避免太多很小的数相乘造成下溢，对乘积取自然对数
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def test_train():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    vec = setOfWords2Vec(myVocabList, listOPosts[0])
    print(vec)

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)

def split_text():
    mySent = 'This is the best book.'
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(mySent)
    print(listOfTokens)
    listOfTokens_ = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    print(listOfTokens_)

def textParse(bigString):
    import re
    listOfTokens = re.split('\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = [];
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='gbk').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='gbk').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 划分训练集和测试集
    trainingSet = list(range(50)); testSet = [];
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    # 训练
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat),array(trainClasses))
    errorCount = 0

    # 测试
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))



def run_main():
    # testingNB()
    # split_text()
    spamTest()
    print()

if __name__ == '__main__':
    run_main()