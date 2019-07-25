# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:48:25 2019

@author: Administrator
"""
    
import numpy as np
import pandas as pd


data = pd.read_excel('成都周边三日.xlsx')
data = data[['price','people','sales']]

# -*- coding: UTF-8 -*-
from math import log,log2
data = data.head(200)
#print(data)

#

#print(data.iloc[:,-1]) 切片
sharemean = data.sales.mean()
readmean = data.people.mean()
pushmean = data.price.mean()
for i in range(len(data.price)):
    if int(data.sales[i]) > sharemean:
        data.sales[i] = 'yes'
    else:
        data.sales[i] = 'no'
    if int(data.people[i]) > readmean   :
        data.people[i] = int(1)
    else:
        data.people[i] = int(0)   
    if data.price[i] > pushmean:
        data.price[i] = int(1)
    else:
        data.price[i] = int(-1)
#labels = ['推送','阅读','分享']
#prob1 = pd.value_counts(data.share) / len(data.share)
##print(prob1)
#shang = sum(np.log2(prob1)*prob1*(-1))
#print(shang)
##
print('计算完毕')
data = data.values.tolist()





#信息增益算法————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def createDataSet(data):#定义结构列表
#    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'yes'], [1, 0, 'yes'], [0, 0, 'no'], [1, 0, 'yes'], [1, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [1, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [1, 0, 'yes'], [0, 0, 'no'], [1, 0, 'yes'], [1, 0, 'yes'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], [1, 0, 'yes'], [1, 0, 'no']]
#    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'yes'], [0, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'yes'], [1, -1, 'yes'], [0, -1, 'no'], [1, -1, 'yes'], [1, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [1, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [1, -1, 'yes'], [0, -1, 'no'], [1, -1, 'yes'], [1, -1, 'yes'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [0, -1, 'no'], [1, -1, 'yes'], [1, -1, 'no']]
#    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'yes'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'yes']]
#    print(np.array(dataSet))
    dataSet = data 
    
    labels = ['price', 'people']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#数据集长度
    labelCounts = {}  #用于统计每个结果的个数
    for featVec in dataSet:  # 添加每列的标签
        currentLabel = featVec[-1]  #数据集结果列
        if currentLabel not in labelCounts.keys(): #
            labelCounts[currentLabel] = 0 #没有的话创建新的标签为0
        labelCounts[currentLabel] += 1 #计算标签数量 ，相同则加一   
    shannonEnt = 0.0  
    for key in labelCounts: 
        prob = float(labelCounts[key]) / numEntries  #每个标签所占概率
        shannonEnt -= prob * log(prob, 2)   # 计算乘积    
    return shannonEnt  #返回指定数据集的信息熵

def splitDataSet(dataSet, axis, value):#划分数据集
#    print(axis,value)
    retDataSet = []  #用于装取每列特征分类后的列表
    for featVec in dataSet :#逐行读取
#        print(featVec)
        if featVec[axis] == value:  # axis为列数
            l = [featVec[axis],featVec[-1]]  #生成每列第n行的特征以及结果
#            reducedFeatVec = featVec[:axis]  #因为列表语法,所以实际上是去除第axis列的内容               
#            reducedFeatVec.extend(featVec[axis + 1:])#扩展列表
#            print(l)
            retDataSet.append(l) #添加列表
#    print(retDataSet)        
    return retDataSet  #返回分类后的每列特征以及结果
def chooseBestFeatureToSp(dataSet,labels):   #选择最好的数据集划分方式
    numFeature = len(dataSet[0])-1    #取出结果列后的列数
    baseEntropy = calcShannonEnt(dataSet)   #整个数据集的信息熵，用于减去单列数据集的信息熵求出信息增益
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet]    #获取每列特征所有的取值列表
#        print(featureList)
        uniqueVals = set(featureList)   #统计每列特征有多少个结果，去重即是结果
        newEntropy = 0.0   
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)   #第i列的数据集特征，value值，划分数据集，代入函数分别统计每个特征的结果
#            print(subDataSet)
            prob = len(subDataSet)/float(len(dataSet))   #数据集特征为i的所占的比例
#            print('概率为%.5f'%prob)
            
            newEntropy +=prob * calcShannonEnt(subDataSet)   #计算每列每种特征以及结果的信息熵
        
        infoGain = baseEntropy- newEntropy  #返回每列特征的信息熵
        print('%s特征的信息增益为%s'%(labels[i],str(infoGain)))
        
        #计算最好的信息增益，增益越大说明所占决策权越大
#        if (infoGain > bestInfoGain):#全部为正
#            bestInfoGain = infoGain
#            bestFeature = i
#    return bestFeature#返回最好的特征
#
dataSet, labels = createDataSet(data)
print('香农熵为:%s'%str(calcShannonEnt(dataSet)))
chooseBestFeatureToSp(dataSet,labels)
























