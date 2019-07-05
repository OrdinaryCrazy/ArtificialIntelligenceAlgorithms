# -*- coding:utf-8 -*-
import  numpy   as      np
import  pandas  as      pd
from    pandas  import  Series, DataFrame
from    collections import  Counter
#------------------------------------------------------------------------------------
def PCA(data, threshold):
    '''
    '''
    covMatrix = np.cov(data, rowvar = 0)
    eigenValues, eigenVectors = np.linalg.eig(np.mat(covMatrix))
    eigenValuesSortedIndex = np.argsort(eigenValues)
    m = 1
    while float(np.sum(eigenValues[eigenValuesSortedIndex[-1:-(m+1):-1]]) / np.sum(eigenValues)) < threshold:
        print(float(np.sum(eigenValues[eigenValuesSortedIndex[-1:-(m+1):-1]]) / np.sum(eigenValues)))
        m += 1
    newData = data * eigenVectors[:, eigenValuesSortedIndex[-1:-(m+1):-1]]
    return newData
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
    diffMatrix = np.tile(point,(np.shape(dataSet)[0],1)) - dataSet
    diffSquarM = diffMatrix.A ** 2
    distanceM  = diffSquarM.sum(axis = 1) ** 0.5
    neighor = np.nonzero(distanceM < eps)[0]
    return neighor
#------------------------------------------------------------------------------------
def DBSCAN(data, label, eps, minPts):
    '''
        密度聚类算法
            输入:
                eps 为邻域 距离阈值
                MinPts 描述了某一样本的距离为 eps 的邻域中样本个数的阈值
    '''
    print("Clustering...")
    # 初始化核心对象集合
    coreObjects = {}
    Clusters = {}
    sampleNum = np.shape(data)[0]
    #找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range(sampleNum) :
        neighbor = getNeighbor(data[i],data,eps)
        if len(neighbor) >= minPts :
            coreObjects[i] = neighbor
    initCoreObjects = coreObjects.copy()
    #初始化聚类簇数
    k = 0
    #初始化未访问样本集合（索引）
    notAccess = list(range(sampleNum))
    print("aggregating...")
    while len(coreObjects) > 0:
        lastNotAccess = []
        lastNotAccess.extend(notAccess)
        cores = list(coreObjects.keys())
        #随机选取一个核心对象
        core = cores[np.random.randint(0, len(cores))]

        queue = []
        queue.append(core)
        notAccess.remove(core)

        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in initCoreObjects.keys() :
                #Δ = N(q)∩Γ 
                delta = [val for val in initCoreObjects[q] if val in notAccess]
                #将Δ中的样本加入队列Q
                queue.extend(delta)
                #Γ = Γ\Δ
                notAccess = [val for val in notAccess if val not in delta]
        k += 1
        Clusters[k] = [val for val in lastNotAccess if val not in notAccess]
        for x in Clusters[k]:
            if x in coreObjects.keys():
                del coreObjects[x]
#----------------------------------------------------------------------
    print(str(k) + " Clusters")
    clusterAssign = np.zeros((sampleNum, 1))
    for c in Clusters.keys() :
        pointsOfThisC = Clusters[c]
        print(len(pointsOfThisC))
        for i in pointsOfThisC :
            clusterAssign[i] = c
    print(clusterAssign)
    rightSum = 0
    for i in range(1, k + 1) :
        labelOfSample = label[np.nonzero(clusterAssign[:,0] == i)[0]]
        labelCount = Counter(labelOfSample)
        # print(labelCount.most_common(1))
        # input()
        rightSum += labelCount.most_common(1)[0][1]
    purity = float(rightSum) / sampleNum
    a = 0
    d = 0
    for i in range(sampleNum) :
        for j in range(sampleNum) :
            if label[i] == label[j] and clusterAssign[i,0] == clusterAssign[j,0] :
                a += 1
            if label[i] != label[j] and clusterAssign[i,0] != clusterAssign[j,0] :
                d += 1
    RI = (a + d) / (sampleNum*sampleNum)
    resultFile = open("DBSCAN.csv",'w')
    resultFile.write(str(k) + "\n")
    for i in range(sampleNum):
        resultFile.write(str(int(clusterAssign[i,0])) + "\n")
    resultFile.close()
    return purity, RI
#------------------------------------------------------------------------------------
inputData = pd.read_csv("../../datasets/青蛙聚类/Frogs_MFCCs.csv")
# 选用 Species 聚类，原来是10个类
resultFile = open("DBSCAN_result.txt",'w')
for eps in range(1,21) :
    newData = PCA(inputData.drop(columns=['Family','Genus','Species','RecordID']).values,0.8)
    for minPts in range(5,15):
    # for k in range(8,13):
        purity, RI = DBSCAN(newData, inputData['Species'].values, float(eps)/10.0, minPts)
        resultFile.write("eps = "       + str(float(eps)/10.0)  + "\t")
        resultFile.write("minPts = "    + str(minPts)           + "\t")
        resultFile.write("purity = "    + str(purity)           + "\t")
        resultFile.write("RI = "        + str(RI)               + "\n")
        print(str(float(eps)/10.0) + "\t" + str(minPts) + "\t" + str(purity) + "\t" + str(RI))
resultFile.close()