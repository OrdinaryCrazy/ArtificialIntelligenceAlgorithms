# -*- coding:utf-8 -*-
import  numpy   as      np
import  pandas  as      pd
from    pandas  import  Series, DataFrame
#==========================================================================================================
# 分类器实现
#==========================================================================================================
class SVMClassifier:
    '''
    '''
#------------------------------------------------------------------------------------
    def __init__(self, trainset, trainlabel, testset, testlabel, sigma = 0, C = 1, toler = 0.001):
        '''
            输入：
                C: soft margin SVM的控制函数
                sigma:  sigma = 0 : 线性核
                        sigma = others: RBF核
        '''
        if(trainset.shape[0] != trainlabel.shape[0]):
            raise TypeError("trainset should have same size with trainlabel.")
        if(testset.shape[0] != testlabel.shape[0]):
            raise TypeError("testset should have same size with testlabel.")
        self.trainX     = np.mat(trainset)
        # self.trainY     = np.mat(trainlabel)
        self.trainY     = trainlabel
        self.testX      = np.mat(testset)
        self.testY      = testlabel
        self.sigma      = sigma
        self.C          = C
        # self.alphas     = np.mat(np.zeros((self.trainX.shape[0],1)))
        self.alphas     = np.zeros(self.trainX.shape[0])
        self.predictY   = []
        self.kernelM    = self.calculateKernelM()
        self.b          = 0
        self.toler      = toler
        self.errorCache = np.mat(np.zeros((self.trainX.shape[0],2)))
#------------------------------------------------------------------------------------
    def calculateError(self, alpha_k):
        '''
            calculate the error for alpha k
        '''
        output_k = float(np.multiply(self.alphas, self.trainY).T * self.kernelM[:, alpha_k] + self.b)
        error_k = output_k - float(self.trainY[alpha_k])
        return error_k
#------------------------------------------------------------------------------------
    def calculateKernelM(self):
        '''
            calculate kernel matrix given train set and kernel type
        '''
        kernelMat = np.mat(np.zeros((self.trainX.shape[0], self.trainX.shape[0])))
        for i in range(self.trainX.shape[0]) :
            kernelMat[:, i] = self.calculateKernelV(i)
        return kernelMat
#------------------------------------------------------------------------------------
    def calculateKernelV(self, i):
        '''
            calulate kernel value
        '''
        kernelVal = np.mat(np.zeros((self.trainX.shape[0], 1)))
        if self.sigma == 0 :
            kernelVal = self.trainX * self.trainX[i, :].T
            # print(kernelVal)
        else :
            for j in range(self.trainX.shape[0]):
                diff = self.trainX[j, :] - self.trainX[i, :]
                kernelVal[j] = np.exp(diff * diff.T / (-1.0 * self.sigma ** 2))
        return kernelVal
#------------------------------------------------------------------------------------
    def selectAlpha_j(self, alpha_i, error_i):
        '''
            select alpha j which has the biggest step
        '''
        self.errorCache[alpha_i] = [1, error_i]
        candidateAlphaList = np.nonzero(self.errorCache[:, 0].A)[0]
        maxStep = 0
        alpha_j = 0
        error_j = 0

        if len(candidateAlphaList) > 1:
            # find the alpha with max iterative step
            for alpha_k in candidateAlphaList:
                if alpha_k == alpha_i:
                    continue
                error_k = self.calculateError(alpha_k)
                if abs(error_k - error_i) > maxStep :
                    maxStep = abs(error_k - error_i)
                    alpha_j = alpha_k
                    error_j = error_k
        else:
            # if came in this loop first time, we select alpha j randomly
            alpha_j = alpha_i
            while alpha_j == alpha_i :
                alpha_j = int(np.random.uniform(0,self.trainX.shape[0]))
            error_j = self.calculateError(alpha_j)
        return alpha_j, error_j
#------------------------------------------------------------------------------------
    def innerLoop(self, alpha_i):
        '''
            the inner loop for optimizing alpha i and alpha j
        '''
        ### check and pick up the alpha who violates the KKT condition
	    ## satisfy KKT condition
	    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
	    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
	    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
	    ## violate KKT condition
	    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
	    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
	    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
	    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
        error_i = self.calculateError(alpha_i)
        if  ( self.trainY[alpha_i] * error_i < - self.toler and self.alphas[alpha_i] < self.C ) \
            or \
            ( self.trainY[alpha_i] * error_i > self.toler and self.alphas[alpha_i] > 0 ) :

            # step 1: select alpha j
            alpha_j, error_j = self.selectAlpha_j(alpha_i, error_i)
            alpha_i_cache = self.alphas[alpha_i].copy()
            alpha_j_cache = self.alphas[alpha_j].copy()
            # step 2: calculate the boundary L and H for alpha j
            if self.trainY[alpha_i] != self.trainY[alpha_j] :
                L = max(0,      0      + self.alphas[alpha_j] - self.alphas[alpha_i])
                H = min(self.C, self.C + self.alphas[alpha_j] - self.alphas[alpha_i])
            else :
                L = max(0, self.alphas[alpha_j] + self.alphas[alpha_i] - self.C)
                H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])
            if L == H :
                return 0
            # step 3: calculate eta (the similarity of sample i and j)
            eta = 2.0 * self.kernelM[alpha_i, alpha_j] - self.kernelM[alpha_i, alpha_i] - self.kernelM[alpha_j, alpha_j]
            if eta >= 0:
                return 0
            # step 4: update alpha j
            self.alphas[alpha_j] -= self.trainY[alpha_j] * (error_i - error_j) / eta
            # step 5: clip alpha j
            self.alphas[alpha_j] = max(L, min(self.alphas[alpha_j], H))
            # step 6: if alpha j not moving enough, just return
            if abs(self.alphas[alpha_j] - alpha_j_cache) < 0.00001:
                self.errorCache[alpha_j] = [1, self.calculateError(alpha_j)]
                return 0
            # step 7: update alpha i after optimizing aipha j
            self.alphas[alpha_i] += self.trainY[alpha_i] * self.trainY[alpha_j] * (alpha_j_cache - self.alphas[alpha_j])
            # step 8: update threshold b
            b1 = self.b - error_i - self.trainY[alpha_i] * (self.alphas[alpha_i] - alpha_i_cache) * self.kernelM[alpha_i, alpha_i] \
                                  - self.trainY[alpha_j] * (self.alphas[alpha_j] - alpha_j_cache) * self.kernelM[alpha_i, alpha_j]
            b2 = self.b - error_j - self.trainY[alpha_i] * (self.alphas[alpha_i] - alpha_i_cache) * self.kernelM[alpha_i, alpha_j] \
                                  - self.trainY[alpha_j] * (self.alphas[alpha_j] - alpha_j_cache) * self.kernelM[alpha_j, alpha_j]
            if      0 < self.alphas[alpha_i] and self.alphas[alpha_i] < self.C :
                self.b = b1
            elif    0 < self.alphas[alpha_j] and self.alphas[alpha_j] < self.C :
                self.b = b2
            else :
                self.b = (b1 + b2) / 2.0
            # step 9: update error cache for alpha i, j after optimize alpha i, j and b
            self.errorCache[alpha_j] = [1, self.calculateError(alpha_j)]
            self.errorCache[alpha_i] = [1, self.calculateError(alpha_i)]

            return 1
        else:
            return 0
#------------------------------------------------------------------------------------
    def trainSVM(self):
        '''
        '''
        entireSet = True
        alphaPairsChanged = 0
        iterCount = 0
        while alphaPairsChanged > 0 or entireSet :
            alphaPairsChanged = 0
            if entireSet:
                # update alphas over all training examples
                for i in range(self.trainX.shape[0]):
                    alphaPairsChanged += self.innerLoop(i)
                print("iter: %d entire set, alpha pairs changed: %d"%(iterCount,alphaPairsChanged))
                iterCount += 1
            else:
                # update alphas over examples where alpha is not 0 & not C (not on boundary)
                nonBoundAlphasList = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in nonBoundAlphasList:
                    alphaPairsChanged += self.innerLoop(i)
                print("iter: %d non boundary, alpha pairs changed: %d"%(iterCount,alphaPairsChanged))
                iterCount += 1
            # alternate loop over all examples and non-boundary examples
            if entireSet :
                entireSet = False
            elif alphaPairsChanged == 0 :
                entireSet = True
#------------------------------------------------------------------------------------
    def testSVM(self):
        '''
        '''
        temp_testX = np.mat(self.testX)
        temp_testY = np.mat(self.testY)
        supportVectorsIndex = np.nonzero(self.alphas > 0)[0]
        supportVectors      = self.trainX[supportVectorsIndex]
        supportVectorsLabel = self.trainY[supportVectorsIndex]
        supportVectorsAlpha = self.alphas[supportVectorsIndex]

        for i in range(self.testX.shape[0]):
            kernelVal = np.mat(np.zeros((supportVectors.shape[0], 1)))
            if self.sigma == 0 :
                kernelVal = supportVectors * self.testX[i, :].T
            else :
                for j in range(supportVectors.shape[0]):
                    diff = supportVectors[j, :] - self.testX[i, :]
                    kernelVal[j] = np.exp(diff * diff.T / (-1.0 * self.sigma ** 2))
            predict = np.multiply(supportVectorsLabel, supportVectorsAlpha) * kernelVal + self.b
            # self.predictY.append(np.sign(predict))
            self.predictY.append(float(predict))
#------------------------------------------------------------------------------------
def computeResult(predictY, targetY):
    '''
        计算评价结果:
        输入:
            无
        输出:
            result{Accuracy, Macro_F1, Micro_F1}
    '''
    result = {  'Accuracy' : 0,     'Macro_F1' : 0,     'Micro_F1' : 0  }
    TP = {  'draw'      : 1,    'eight'     : 1,    'eleven'    : 1,    'fifteen'   : 1,    'five'  : 1,    'four'      : 1,
            'fourteen'  : 1,    'nine'      : 1,    'one'       : 1,    'seven'     : 1,    'six'   : 1,    'sixteen'   : 1,
            'ten'       : 1,    'thirteen'  : 1,    'three'     : 1,    'twelve'    : 1,    'two'   : 1,    'zero'      : 1
            }
    FP = TP.copy()
    TN = TP.copy()
    FN = TP.copy()
    for key in TP :
        for i in range(targetY.shape[0]) :
            if targetY[i] == key and predictY[i] == key :
                TP[key] = TP[key] + 1
            if targetY[i] == key and predictY[i] != key :
                FN[key] = FN[key] + 1
            if targetY[i] != key and predictY[i] == key :
                FP[key] = FP[key] + 1
            if targetY[i] != key and predictY[i] == key :
                TN[key] = TN[key] + 1
    TPsum = 0
    TNsum = 0
    FPsum = 0
    FNsum = 0
    F1sum = 0
    for key in TP :
        TPsum = TPsum + TP[key]
        TNsum = TNsum + TN[key]
        FPsum = FPsum + FP[key]
        FNsum = FNsum + FN[key]
        tempPrecision = TP[key] / (TP[key] + FP[key])
        tempRecall = TP[key] / (TP[key] + FN[key])
        F1sum = F1sum + (2 * tempPrecision * tempRecall) / (tempPrecision + tempRecall)
    result['Accuracy'] = (TPsum + TNsum) / (TPsum + TNsum + FPsum + TNsum)
    result['Macro_F1'] = F1sum / len(TP)
    Micro_Precision = TPsum / (TPsum + FPsum)
    Micro_Recall = TPsum / (TPsum + FNsum)
    result['Micro_F1'] = (2 * Micro_Precision * Micro_Recall) / (Micro_Precision + Micro_Recall)
    print(result)
    return result
#==========================================================================================================
# 实验要求接口包装
#==========================================================================================================
def multiClassSVM(trainset, trainlabel, testset, testlabel, sigma = 0, C = 1):
    '''

    '''
    # svmKinds = DataFrame(testlabel)[0].unique()
    svmKinds = trainlabel.unique()
    probList = [[] for label in testlabel]
    for kind in svmKinds:
        print(kind)
        temptestY  = testlabel == kind
        temptestY  = temptestY.replace([True,False],[1,-1])
        temptrainY = trainlabel == kind
        temptrainY = temptrainY.replace([True,False],[1,-1])

        thisKindPredictY = softSVM(trainset, temptrainY.values, testset, temptestY.values, sigma, C)

        for i in range(len(testlabel)) :
            probList[i].append(thisKindPredictY[i])

    allpredict = []
    for i in range(testlabel.shape[0]) :
        allpredict.append(svmKinds[probList[i].index(max(probList[i]))])
    
    result = computeResult(allpredict,testlabel)
    return result, allpredict
#------------------------------------------------------------------------------------
def softSVM(trainset, trainlabel, testset, testlabel, sigma, C):
    '''

    '''
    svmc = SVMClassifier(trainset, trainlabel, testset, testlabel, sigma, C)
    svmc.trainSVM()
    svmc.testSVM()
    return svmc.predictY
#==========================================================================================================
# 测试统计
#==========================================================================================================
originFeature = ['WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR', 'winstep']

trainset = pd.read_csv( "../../datasets/国际象棋Checkmate预测/trainset.csv" , 
                        header = 0                  ,
                        names  = originFeature
                        )
trainset[['WKC','WRC','BKC']] = trainset[['WKC','WRC','BKC']].replace(['a','b','c','d','e','f','g','h'],[1,2,3,4,5,6,7,8])
trainset['WKC_WRC'] = abs(trainset['WKC'] - trainset['WRC'])
trainset['WKC_BKC'] = abs(trainset['WKC'] - trainset['BKC'])
trainset['WRC_BKC'] = abs(trainset['WRC'] - trainset['BKC'])
trainset['WKR_WRR'] = abs(trainset['WKR'] - trainset['WRR'])
trainset['WKR_BKR'] = abs(trainset['WKR'] - trainset['BKR'])
trainset['WRR_BKR'] = abs(trainset['WRR'] - trainset['BKR'])
trainset['ATTRACK'] = trainset['WRC_BKC'] + trainset['WRR_BKR']

testset  = pd.read_csv( "../../datasets/国际象棋Checkmate预测/testset.csv"  ,  
                        header  = 0                 ,
                        names   = originFeature
                        )
testset[['WKC','WRC','BKC']] = testset[['WKC','WRC','BKC']].replace(['a','b','c','d','e','f','g','h'],[1,2,3,4,5,6,7,8])
testset['WKC_WRC'] = abs(testset['WKC'] - testset['WRC'])
testset['WKC_BKC'] = abs(testset['WKC'] - testset['BKC'])
testset['WRC_BKC'] = abs(testset['WRC'] - testset['BKC'])
testset['WKR_WRR'] = abs(testset['WKR'] - testset['WRR'])
testset['WKR_BKR'] = abs(testset['WKR'] - testset['BKR'])
testset['WRR_BKR'] = abs(testset['WRR'] - testset['BKR'])
testset['ATTRACK'] = testset['WRC_BKC'] + testset['WRR_BKR']

# resultFile = open("SVMresult.txt",'w')

# trainset = trainset.sample(n = 800)
# for i in range(4):
#     result, ypred = multiClassSVM(  trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']].values,
#                                     trainset['winstep'],
#                                     testset[[  'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']].values,
#                                     testset[ 'winstep'],
#                                     sigma = i
#                                     )
#     resultFile.write("sigma = " + str(i) + "\t")
#     resultFile.write("Accuracy = " + str(result['Accuracy']) + "\t")
#     resultFile.write("Macro_F1 = " + str(result['Macro_F1']) + "\t")
#     resultFile.write("Micro_F1 = " + str(result['Micro_F1']) + "\n")

# resultFile.close()

resultFile = open("SVMresultS.txt",'w')

trainset = trainset.sample(n = 800)
result, ypred = multiClassSVM(  trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']].values,
                                trainset['winstep'],
                                testset[[  'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']].values,
                                testset[ 'winstep'],
                                sigma = 1.6,
                                C = 10
                                )
resultFile.write("sigma = " + str(1.6) + "\t")
resultFile.write("C = " + str(10) + "\t")
resultFile.write("Accuracy = " + str(result['Accuracy']) + "\t")
resultFile.write("Macro_F1 = " + str(result['Macro_F1']) + "\t")
resultFile.write("Micro_F1 = " + str(result['Micro_F1']) + "\n")

resultFile.close()

