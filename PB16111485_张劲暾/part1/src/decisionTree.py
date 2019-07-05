# -*- coding:utf-8 -*-
import  numpy               as      np
import  pandas              as      pd
import  matplotlib.pyplot   as      plt
import  math
from    pandas  import  Series, DataFrame
# import treeplot
#==========================================================================================================
# 分类器实现
#==========================================================================================================
class ID3Classifier:
    '''
        Krkopt 残局预测 ID3决策树算法预测器
        张劲暾 2019.07.01
    '''
    originFeature = ['WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR', 'winstep']
#------------------------------------------------------------------------------------
    def __init__(self, trainset, trainlabel, testset, testlabel, threshold = 0.0):
        '''
            初始化参数说明：(输入输出类型均为pandas.DataFrame)
                输入:
                    trainset    : 训练集数据
                    trainlabel  : 训练集标签
                    testset     : 测试集数据
                    testlabel   : 测试集标签
                    threshold   : 划分阈值
                输出:
                    无
        '''
        if(trainset.shape[0] != trainlabel.shape[0]):
            raise TypeError("trainset should have same size with trainlabel.")
        if(testset.shape[0] != testlabel.shape[0]):
            raise TypeError("testset should have same size with testlabel.")
        self.trainX = trainset
        self.trainY = trainlabel
        self.testX  = testset
        self.testY  = testlabel
        self.threshold = threshold
        self.predictY = []
#------------------------------------------------------------------------------------
    def calculateShannonEnt(self, dataSetX, dataSetY):
        '''
            计算信息熵
                输入:
                    dataSetX:数据集特征向量
                    dataSetY:数据集标签
                输出:
                    信息熵
        '''
        numEntries = len(dataSetX)
        labelCount = {}
        for label in dataSetY :
            labelCount[label] = labelCount.get(label,0) + 1
        shannonEnt = 0.0
        for key in labelCount :
            probility = float(labelCount[key]) / numEntries
            shannonEnt = shannonEnt - probility * math.log2(1/probility)
        return shannonEnt
#------------------------------------------------------------------------------------
    def calculateConditionE(self, dataSetX, dataSetY, axis):
        '''
            计算条件熵
                输入:(输入类型为pandas.DataFrame)
                    dataSetX:数据集特征向量
                    dataSetY:数据集标签
                    axis:计算条件熵的列(列名index)
                输出:
                    axis这一列对应的条件熵
        '''
        condiEnt = 0.0
        # featureKind = DataFrame(pd.concat([dataSetX,dataSetY],axis=1), columns=self.originFeature)[axis].unique()
        featureKind = dataSetX[axis].unique()
        for keyValue in featureKind :
            # tempS = DataFrame(pd.concat([dataSetX,dataSetY],axis=1), columns=self.originFeature)
            tempS = pd.concat([dataSetX,dataSetY],axis=1)
            subS = tempS.loc[tempS[axis] == keyValue]
            proportion = float(len(subS)) / len(tempS)
            condiEnt = condiEnt + proportion * self.calculateShannonEnt(subS.drop(columns='winstep').values,subS['winstep'].values)
        return condiEnt
#------------------------------------------------------------------------------------
    def chooseBestFeature(self, datasetX, dataSetY):
        '''
            利用信息熵实现选取特征，划分数据集，计算得到当前最好的划分数据集的特征
                输入:
                    dataSetX:数据集特征向量
                    dataSetY:数据集标签
                输出:
                    选取的最佳维度
        '''
        baseEnt = self.calculateShannonEnt(datasetX, dataSetY)
        bestIG = 0.0
        bestFeature = None
        for i in range(datasetX.shape[1]):
            IG = self.calculateConditionE(datasetX, dataSetY, self.originFeature[i]) - baseEnt
            if IG > bestIG:
                bestIG = IG
                bestFeature = self.originFeature[i]
        # print(bestIG)
        return bestFeature, bestIG
#------------------------------------------------------------------------------------
    def createTree(self):
        '''
            建树预测
                输入:
                    无
                输出:
                    result = describe{Accuracy, Macro_F1, Micro_F1}
                    统计结果和预测结果
                    图形化决策树
        '''
        ID3DecisionTree = self.createSubTree(self.trainX, self.trainY, level = 0)
        for i in range(self.testX.shape[0]):
            self.predictY.append(self.classify(ID3DecisionTree, self.testX.iloc[i]))
        describe = self.computeResult();
        # self.createPlot(ID3DecisionTree)
        # treeplot.createPlot(ID3DecisionTree)
        print("painting......")
        createPlot(ID3DecisionTree, self.threshold)
        print("Figure get!")
        return describe
#------------------------------------------------------------------------------------
    def classify(self, searchTree, testVec):
        '''
            分类函数
                输入:
                    searchTree:用于分类的树
                    testVec:要分类的特征向量
                输出:
                    预测的类别
        '''
        if type(searchTree).__name__ == 'str':
            return searchTree
        firstKey = list(searchTree.keys())[0]
        sendDict = searchTree[firstKey]
        for key in sendDict.keys() :
            if testVec[firstKey] == key:
                if type(sendDict[key]).__name__ == 'dict':
                    return self.classify(sendDict[key], testVec)
                else :
                    return sendDict[key]
#------------------------------------------------------------------------------------
    def createSubTree(self, dataSetX, dataSetY, level):
        '''
            构建子树
                输入:(输入类型为pandas.DataFrame)
                    dataSetX:数据集特征向量
                    dataSetY:数据集标签
                输出:
                    子树
        '''
        labelList = list(dataSetY)
        # 只有一个类就可以返回了
        if labelList.count(labelList[0]) == len(labelList) :
            return labelList[0]
        # 没有特征可以选择就返回最多的那一类
        if dataSetX.shape[1] == 0 :
            return self.majorityCount(labelList)
        bestF, bestIG = self.chooseBestFeature(dataSetX,dataSetY)
        if bestIG < self.threshold and  level > 2:
            return self.majorityCount(labelList)
        subTree = {bestF:{}}
        featureKind = dataSetX[bestF].unique()
        tempS = pd.concat([dataSetX, dataSetY],axis=1)
        for feature in featureKind :
            subS = tempS.loc[tempS[bestF] == feature]
            subTree[bestF][feature] = self.createSubTree(subS.drop(columns='winstep'), subS['winstep'], level + 1)
        return subTree
#------------------------------------------------------------------------------------
    def majorityCount(self, dataSetY):
        '''
            选出出现次数最多的类别
                输入:
                    dataSetY:数据集标签
                输出:
                    dataSetY中最多的标签
        '''
        labelCount = {}
        for label in dataSetY :
            labelCount[label] = labelCount.get(label,0) + 1
        labelCountSorted = sorted(labelCount.items(), key = lambda x:x[1], reverse = True)
        return labelCountSorted[0][0]
#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
# 定义文本框和箭头格式
# 判断节点样式
decisionNode   = dict(boxstyle="round4", color='#3366FF')
# 叶子节点样式
leafNode       = dict(boxstyle="circle", color='#FF6633')
# 箭头样式
arrowArgs      = dict(arrowstyle="<-", color='g')
#------------------------------------------------------------------------------------
def plotNode(nodeText, centerPt, parentPt, nodeType):
    '''
        绘制带箭头的注释
        centerPt: 节点中心坐标  
        parentPt: 起点坐标
    '''
    createPlot.ax1.annotate(    nodeText,
                                xy=parentPt,
                                xycoords='axes fraction',
                                xytext=centerPt,
                                textcoords='axes fraction',
                                va='center',
                                ha='center',
                                bbox=nodeType,
                                arrowprops=arrowArgs
                                )
#------------------------------------------------------------------------------------
def createPlot(tree, threshold):
    '''
        创建新图形并清空绘图区
            输入:
                tree:要绘制的决策树
            输出:
                决策树图形
    '''
    # figure = plt.figure(1,facecolor='white',figsize=(200,200))
    figure = plt.figure(1,facecolor='white')
    figure.clf()

    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon = False,**axprops)
    # 创建两个全局变量存储树的宽度和深度
    # plotTree.totalW = 50000.0 * float(getNumLeafs(tree))
    # plotTree.totalD =    50.0 * float(getTreeDepth(tree))

    # 算了，放大了也看不清，pdf很难打开，倒不如小一点，起码能看清六层逻辑结构
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))

    # 追踪已经绘制的节点位置,初始值为将总宽度平分,在取第一个的一半 
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    # 调用函数，并指出根节点源坐标 
    plotTree(tree,(0.5,1.0),'')

    # plt.savefig("./ID3DecisionTree" + str(threshold) + ".eps",format='eps', dpi=1000)
    plt.savefig("./ID3DecisionTree" + str(threshold) + ".png")
#------------------------------------------------------------------------------------
def plotMidText(centerPt, parentPt, txtString):
    '''
        在父子结点间填充文本信息
    '''
    xMid = (parentPt[0]-centerPt[0])/2.0 + centerPt[0]
    yMid = (parentPt[1]-centerPt[1])/2.0 + centerPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
#------------------------------------------------------------------------------------
def plotTree(tree, parentPt, nodeText):
    '''
        计算叶子数量
    '''
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)
    # 定位
    # center = (plotTree.xOff + (100000.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    center = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)

    if type(tree).__name__ == 'str':
        plotTree.xOff += 1.0/plotTree.totalW
        plotNode(tree,(plotTree.xOff,plotTree.yOff),center,leafNode)
        plotMidText((plotTree.xOff,plotTree.yOff),center,tree)
        return
    
    firstStr = list(tree.keys())[0]
    # 中间的文本
    plotMidText(center,parentPt,nodeText)
    # 节点
    plotNode(firstStr,center,parentPt,decisionNode)
    secondDict = tree[firstStr]
    # 减少y的值，将树的总深度平分，每次减少移动一点(向下，因为树是自顶向下画的）
    plotTree.yOff -= 1.0/plotTree.totalD
    # plotTree.yOff -= 50.0/plotTree.totalD
    # 开始画了，也是递归
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],center,str(key))
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            # plotTree.xOff += 50000.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),center,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),center,str(key))
    plotTree.yOff += 1.0/plotTree.totalD
    # plotTree.yOff += 50.0/plotTree.totalD
#------------------------------------------------------------------------------------
def getNumLeafs(tree):
    """
        计算叶结点数
    """
    if type(tree).__name__ == 'str':
        return 1
    numLeafs = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        # 如果节点还是一个字典，就说明还可以继续
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            # 每次发现一个节点就加一，最终的那个子叶也是加个1就跑了
            numLeafs += 1
    return numLeafs
#------------------------------------------------------------------------------------
def getTreeDepth(tree):
    '''
        计算树的层数
    '''
    if type(tree).__name__ == 'str':
        return 1
    maxDepth = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
#==========================================================================================================
# 实验要求接口包装
#==========================================================================================================
def chooseBestFeatrure(dataset):
    # 这个只是说明用法，因为每次dataset的feature不一定是这六个全都有
    id3c = ID3Classifier(dataset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']],dataset['winstep'], None, None)
    return id3c.chooseBestFeature(dataset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']],dataset['winstep'])
def createTree(trainset, trainlabel, testset, testlabel):
    id3c = ID3Classifier(trainset, trainlabel, testset, testlabel)
    result = id3c.createTree()
    ypred = id3c.predictY
    return result, ypred
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

resultFile = open("ID3result.txt",'w')
for i in range(0,20):
    id3c = ID3Classifier(   trainset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']],
                            trainset['winstep'],
                            testset[[ 'WKC', 'WKR', 'WRC', 'WRR', 'BKC', 'BKR']],
                            testset['winstep'],
                            float(i) / 10
                            )
    result = id3c.createTree()
    ypred = id3c.predictY
    resultFile.write("threshold = " + str(float(i) / 10) + "\t")
    resultFile.write("Accuracy = " + str(result['Accuracy']) + "\t")
    resultFile.write("Macro_F1 = " + str(result['Macro_F1']) + "\t")
    resultFile.write("Micro_F1 = " + str(result['Micro_F1']) + "\n")
resultFile.close()