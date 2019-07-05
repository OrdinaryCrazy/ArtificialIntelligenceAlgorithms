# -*- coding:utf-8 -*-
import  numpy               as  np
import  pandas              as  pd
import  matplotlib.pyplot   as  plt
from    pandas  import  Series, DataFrame
from    collections import  Counter
#------------------------------------------------------------------------------------
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
inputData = pd.read_csv("../../datasets/青蛙聚类/Frogs_MFCCs.csv")

fig_data = inputData.drop(columns=['Family','Genus','Species','RecordID']).values
fig_covMatrix = np.cov(fig_data, rowvar = 0)
fig_eigenValues, fig_eigenVectors = np.linalg.eig(np.mat(fig_covMatrix))
fig_eigenValuesSortedIndex = np.argsort(fig_eigenValues)
fig_newData = fig_data * fig_eigenVectors[:, fig_eigenValuesSortedIndex[-1:-(2+1):-1]]
# print(fig_newData[:,0].T.A[0])
# figure = plt.figure(1,facecolor='white',figsize=(200,200))
plt.scatter(fig_newData[:,0].T.A[0], fig_newData[:,1].T.A[0], s=10, c='r', alpha=1, lw=0)
plt.savefig("./PCA.png")

# 选用 Species 聚类，原来是10个类
resultFile = open("Kmeans_PCA_result.txt",'a')
for t in range(8,10) :
    newData = PCA(inputData.drop(columns=['Family','Genus','Species','RecordID']).values,float(t)/10.0)
    for k in range(1,21):
    # for k in range(8,13):
        purity, RI = KMeans(newData, inputData['Species'].values, k)
        resultFile.write("k = "         + str(k)                + "\t")
        resultFile.write("threshold = " + str(float(t)/10.0)    + "\t")
        resultFile.write("purity = "    + str(purity)           + "\t")
        resultFile.write("RI = "        + str(RI)               + "\n")
        print(str(k) + "\t" + str(float(t)/10.0) + "\t" + str(purity) + "\t" + str(RI))
resultFile.close()