# -*- coding:utf-8 -*-
import  numpy   as      np
import  pandas  as      pd
from    pandas  import  Series, DataFrame
#==========================================================================================================
# 分类器实现
#==========================================================================================================
class KNNClassifier:
    '''
        Krkopt 残局预测 K近邻算法预测器
        张劲暾 2019.06.29
    '''
    # classWeight = { 'draw'      : 1,    'eight'     : 2,    'eleven'    : 1,
    #                 'fifteen'   : 2,    'five'      : 4,    'four'      : 5,
    #                 'fourteen'  : 1,    'nine'      : 2,    'one'       : 5,
    #                 'seven'     : 4,    'six'       : 4,    'sixteen'   : 4,
    #                 'ten'       : 2,    'thirteen'  : 1,    'three'     : 5,
    #                 'twelve'    : 1,    'two'       : 4,    'zero'      : 5
    #                 }
    classWeight = { 'draw'      : 1,    'eight'     : 1,    'eleven'    : 1,
                    'fifteen'   : 1,    'five'      : 1,    'four'      : 1,
                    'fourteen'  : 1,    'nine'      : 1,    'one'       : 1,
                    'seven'     : 1,    'six'       : 1,    'sixteen'   : 1,
                    'ten'       : 1,    'thirteen'  : 1,    'three'     : 1,
                    'twelve'    : 1,    'two'       : 1,    'zero'      : 1
                    }
    def __init__(self, trainset, trainlabel, testset, testlabel, k = 1):
        '''
            初始化参数说明：(输入输出类型均为numpy.array)
                输入:
                    trainset    : 训练集数据
                    trainlabel  : 训练集标签
                    testset     : 测试集数据
                    testlabel   : 测试集标签
                    k           : 选取的最近邻个数
                输出:
                    无
        '''
        if(trainset.shape[0] != trainlabel.shape[0]):
            raise TypeError("trainset should have same size with trainlabel.")
        if(testset.shape[0] != testlabel.shape[0]):
            raise TypeError("testset should have same size with testlabel.")
        self.k      = k
        self.trainX = trainset
        self.trainY = trainlabel
        self.testX  = testset
        self.testY  = testlabel
        self.predictY = []

    def predict(self):
        '''
            预测函数: 按参数预测并将结果保存在self.predictY
            输入:
                无
            输出:
                self.predictY
        '''
        for i in range(self.testX.shape[0]):
            # 计算到各点的距离并排序
            diffMatrix = np.tile(self.testX[i],(self.trainX.shape[0],1)) - self.trainX
            # diffSquarM = diffMatrix ** 2
            diffSquarM = abs(diffMatrix) * 10
            # distanceM  = diffSquarM.sum(axis = 1) ** 0.5
            distanceM  = diffSquarM.sum(axis = 1)
            distanceMSorted = distanceM.argsort()
            # 选取最近的样本点
            neighborYCount = {}
            for j in range(1, self.k + 1):
                neighborY = self.trainY[distanceMSorted[j]]
                neighborYCount[neighborY] = neighborYCount.get(neighborY,0) + self.classWeight[neighborY] / (1 + distanceM[distanceMSorted[j]])
            neighborYCountSorted = sorted(neighborYCount.items(), key = lambda x:x[1], reverse = True)
            self.predictY.append(neighborYCountSorted[0][0])

    def computeResult(self):
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
            for i in range(self.testY.shape[0]) :
                if self.testY[i] == key and self.predictY[i] == key :
                    TP[key] = TP[key] + 1
                if self.testY[i] == key and self.predictY[i] != key :
                    FN[key] = FN[key] + 1
                if self.testY[i] != key and self.predictY[i] == key :
                    FP[key] = FP[key] + 1
                if self.testY[i] != key and self.predictY[i] == key :
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
def knn(trainset, trainlabel, testset, testlabel, k):
    knnc = KNNClassifier(np.array(trainset), np.array(trainlabel), np.array(testset), np.array(testlabel), k)
    knnc.predict()
    return knnc.predictY
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

resultFile = open("KNNresult.txt",'w')

for i in range(1,12):
    knnc = KNNClassifier(   np.array(trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']]),
    # knnc = KNNClassifier(   np.array(trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR',
                                                # 'WKC_WRC','WKC_BKC','WRC_BKC','WKR_WRR','WKR_BKR','WRR_BKR',
                                                # 'ATTRACK'
                                                # ]]),
                            np.array(trainset['winstep']),
                            np.array(trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']]),
                            # np.array(trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR',
                                                # 'WKC_WRC','WKC_BKC','WRC_BKC','WKR_WRR','WKR_BKR','WRR_BKR',
                                                # 'ATTRACK']]),
                            np.array(trainset['winstep']),
                            k = i
                            )
    knnc.predict()
    result = knnc.computeResult()
    resultFile.write("k = " + str(i) + "\t")
    resultFile.write("Accuracy = " + str(result['Accuracy']) + "\t")
    resultFile.write("Macro_F1 = " + str(result['Macro_F1']) + "\t")
    resultFile.write("Micro_F1 = " + str(result['Micro_F1']) + "\n")

resultFile.close()