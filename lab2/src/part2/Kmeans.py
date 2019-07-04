# -*- coding:utf-8 -*-
import  numpy   as      np
import  pandas  as      pd
from    pandas      import  Series, DataFrame
from    collections import  Counter

maxIter = 50
#------------------------------------------------------------------------------------
def calculateEuclDistance(vector1, vector2):
    '''
        计算两个向量之间的欧式距离
    '''
    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))
#------------------------------------------------------------------------------------
def initCenters(data, k):
    '''
    '''
    sampleNum, dimension = data.shape
    centers = np.zeros((k, dimension))
    for i in range(k):
        index = int(np.random.uniform(0, sampleNum))
        centers[i, :] = data[index, :]
    return centers
#------------------------------------------------------------------------------------
def KMeans(data, label, k = 5):
    '''

    '''
    sampleNum = data.shape[0]
    # first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
    clusterAssign = np.zeros((sampleNum, 2))
    clusterChange = True
    iterCount = 0
    ## step 1: 随机初始化中心点
    centers = initCenters(data, k)

    while iterCount < maxIter and clusterChange:
        clusterChange = False
        ## step 2: 重新分配
        for i in range(sampleNum) :
            minDistance = 1e8
            minIndex = 0
            for j in range(k) :
                distance = calculateEuclDistance(centers[j,:], data[i, :])
                if distance < minDistance :
                    minDistance = distance
                    minIndex = j
                if clusterAssign[i,0] != minIndex:
                    clusterChange = True
                    clusterAssign[i, :] = minIndex, minDistance ** 2
        ## step 3: 重新确定中心点
        for j in range(k) :
            sampleOfCluster = data[np.nonzero(clusterAssign[:,0] == j)[0]]
            centers[j, :] = np.mean(sampleOfCluster, axis=0)
        iterCount += 1
        # print(iterCount)
    print("Cluster finished.")
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
            # if label[i] == label[j] and clusterAssign[i,0] != clusterAssign[j,0] :
            #     b += 1
            # if label[i] != label[j] and clusterAssign[i,0] == clusterAssign[j,0] :
            #     c += 1
            if label[i] != label[j] and clusterAssign[i,0] != clusterAssign[j,0] :
                d += 1
    RI = (a + d) / (sampleNum*sampleNum)
    # print(RI)
    resultFile = open("KMeans.csv",'w')
    resultFile.write(str(k) + "\n")
    for i in range(sampleNum):
        resultFile.write(str(int(clusterAssign[i,0])) + "\n")
    resultFile.close()
    return purity, RI
#------------------------------------------------------------------------------------
inputData = pd.read_csv("../../datasets/青蛙聚类/Frogs_MFCCs.csv")
# 选用 Species 聚类，原来是10个类
resultFile = open("Kmeans_result.txt",'w')
for k in range(1,21):
    purity, RI = KMeans(inputData.drop(columns=['Family','Genus','Species','RecordID']).values, inputData['Species'].values, k)
    resultFile.write("k = "         + str(k)        + "\t")
    resultFile.write("purity = "    + str(purity)   + "\t")
    resultFile.write("RI = "        + str(RI)       + "\n")
    print(str(k) + "\t" + str(purity) + "\t" + str(RI))
resultFile.close()