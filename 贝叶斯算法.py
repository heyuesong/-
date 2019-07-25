# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:23:33 2019

@author: Administrator
"""

from numpy import *
#贝叶斯算法

def loadDataSet():
    trainData=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels=[0, 1, 0, 1, 0, 1] #1表示侮辱性言论，0表示正常言论
    return trainData, labels

#生成词汇表
def createVocabList(trainData):
    VocabList = set([])
    for item in trainData:
        VocabList = VocabList|set(item) #取两个集合的并集
    return sorted(list(VocabList))    #对结果排序后返回

#对训练数据生成只包含0和1的向量集
def createWordSet(VocabList, trainData):
    VocabList_len = len(VocabList)   #词汇集的长度    32
    trainData_len = len(trainData)   #训练数据的长度    6
    WordSet = zeros((trainData_len,VocabList_len))     #生成行长度为训练数据的长度 列长度为词汇集的长度的列表
    for index in range(0,trainData_len):
        for word in trainData[index]:
            if word in VocabList:     #其实也就是，训练数据包含的单词对应的位置为1其他为0
                WordSet[index][VocabList.index(word)] = 1
    return WordSet

#计算向量集每个的概率
def opreationProbability(WordSet, labels):
       WordSet_col = len(WordSet[0]) #32
       labels_len = len(labels) #6
       WordSet_labels_0 = zeros(WordSet_col)  #array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
       WordSet_labels_1 = zeros(WordSet_col)
       num_labels_0 = 0
       num_labels_1 = 0
       for index in range(0,labels_len):
           if labels[index] == 0:
               WordSet_labels_0 += WordSet[index]       #向量相加
               num_labels_0 += 1                        #计数
           else:
               WordSet_labels_1 += WordSet[index]       #向量相加
               num_labels_1 += 1                        #计数
       p0 = WordSet_labels_0 * num_labels_0 / labels_len
       
       p1 = WordSet_labels_1 * num_labels_1 / labels_len
       return p0, p1



trainData, labels = loadDataSet()
VocabList = createVocabList(trainData)
train_WordSet = createWordSet(VocabList,trainData)
p0, p1 = opreationProbability(train_WordSet, labels)
print(p0)
print(p1)
#到此就算是训练完成
#开始测试
testData = [['not', 'take', 'ate', 'my', 'stupid','help']]     #测试数据

test_WordSet = createWordSet(VocabList, testData)      #测试数据的向量集
print(test_WordSet)
res_test_0 = []
res_test_1 = []

for index in range(0,len(p0)):
    if test_WordSet[0][index] == 0:
        res_test_0.append((1-p0[index]) * test_WordSet[0][index])
        res_test_1.append((1-p1[index]) * test_WordSet[0][index])
    else:
        res_test_0.append(p0[index] * test_WordSet[0][index])
        res_test_1.append(p1[index] * test_WordSet[0][index])



if sum(res_test_0) > sum(res_test_1):
    print("属于0类别")
else:
    print("属于1类别")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    