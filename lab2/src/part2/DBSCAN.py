# -*- coding:utf-8 -*-
import  numpy   as      np
import  pandas  as      pd
from    pandas  import  Series, DataFrame
from    collections import  Counter
#------------------------------------------------------------------------------------
def calculateEuclDistance(vector1, vector2):
    '''
        计算两个向量之间的欧式距离
    '''
    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))
#------------------------------------------------------------------------------------
def getNeighbor(point, dataSet, eps):
    '''
        获取一个点的ε-邻域(记录的是索引)
    '''
    neighor = []
    for i in range(np.shape(dataSet)[0]) :
        if calculateEuclDistance(point, dataSet[i]) < eps :
            neighor.append(i)
    return i
#------------------------------------------------------------------------------------
def DBSCAN(data, labels, eps, minPts):
    '''
        密度聚类算法
            输入:
                eps 为邻域 距离阈值
                MinPts 描述了某一样本的距离为 eps 的邻域中样本个数的阈值
    '''
    # 初始化核心对象集合
    coreObjects = {}
    Clusters = {}
    sampleNum = np.shape(data)[0]
    #找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range() :
        neighbor = getNeighbor()
        if len() >= minPts :
            coreObjects[i] = neighbor
    initCoreObjects = coreObjects.copy()
    #初始化聚类簇数

    #初始化未访问样本集合（索引）

    clusterAssign = np.zeros((sampleNum, 1))
    rightSum = 0
    for i in range(k) :
        labelOfSample = label[np.nonzero(clusterAssign[:,0] == i)[0]]
        labelCount = Counter(labelOfSample)
        rightSum += labelCount.most_common(1)[0][1]
    purity = float(rightSum) / sampleNum
    # print(purity)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(sampleNum) :
        for j in range(sampleNum) :
            if label[i] == label[j] and clusterAssign[i,0] == clusterAssign[j,0] :
                a += 1
            if label[i] != label[j] and clusterAssign[i,0] != clusterAssign[j,0] :
                d += 1
    RI = (a + d) / (sampleNum*sampleNum)
    # print(RI)
    resultFile = open("KMeans_PCA.csv",'w')
    resultFile.write(str(k) + "\n")
    for i in range(sampleNum):
        resultFile.write(str(int(clusterAssign[i,0])) + "\n")
    resultFile.close()
    return purity, RI
#------------------------------------------------------------------------------------
inputData = pd.read_csv("../../datasets/青蛙聚类/Frogs_MFCCs.csv")
# 选用 Species 聚类，原来是10个类
resultFile = open("DBSCAN_result.txt",'w')

resultFile.close()