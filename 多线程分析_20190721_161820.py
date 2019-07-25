# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:00:17 2019

@author: Administrator
"""

import pandas as pd

import jieba.analyse
#import wordcloud
from collections import Counter
titlename = '成都周边一日'
from itertools import product

datamsp = pd.read_excel(titlename+'.xlsx',sheet_name='Sheet1')

##查看数据维度
#datamsp.shape
#datamsp = pd.read_excel('青海.xlsx',sheet_name='Sheet1')
#分析数据缺失值h
#import missingno as msno
#msno.bar(datamsp.sample(len(datamsp)),figsize=(10,4))

#删除重复值
datamsp = datamsp.drop_duplicates()
datamsp.shape

#取出5个维度的数据
data = datamsp[['title','price','people','sales','traffic','time']]
data.head(12)

#print(data['title'])
#标题转换为列表
title = data.title.values.tolist()
#jieba分词
import jieba
title_s = []
#jieba.add_word('1日游',tag='d')
for line in title:
    title_cut = jieba.lcut(line)
    title_s.append(title_cut)
#print(title_s)
#********************************************************************************************
#导入停用词表
#路径
#D:\Users\Administrator\AppData\Local\Kingsoft\WPS Office\11.1.0.8597\office6\data\chinesesegment\dict
stopword  = [line.strip() for line in open('C:/Users/Administrator/Desktop/stopwords.txt','r',encoding = 'utf-8').readlines()]
#********************************************************************************************
#剔除停用词
title_clean = []
for line in title_s:
    line_clean = []
    for word in line:
        if word not in stopword:
            line_clean.append(word)
    title_clean.append(line_clean)
#print(title_clean)
    #********************************************************************************************
#进行去重
title_clean_dist = []
for line in title_clean:
    line_dist = []
    for word in line:
        if word not in line_dist:
            line_dist.append(word)
    title_clean_dist.append(line_dist)       
#print(title_clean_dist)
    #********************************************************************************************
#转到一个列表内
allwords_clean_dist = []
for line in title_clean_dist:
    for word in line:
        allwords_clean_dist.append(word)        
#print(allwords_clean_dist)
        
#********************************************************************************************
#转为数据框
df_allwords_clean_dist = pd.DataFrame({
        'allwords':allwords_clean_dist
        })
#print(df_allwords_clean_dist)
    
#分类汇总
word_count = df_allwords_clean_dist.allwords.value_counts().reset_index()
word_count.columns = {'word','count'}
#print(word_count)
L = []
for x in range(15):
    L.append(word_count.word[x])
#print(L)
#********************************************************************************************
#对分组数据词云可视化
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
##from scipy.misc import imread
#plt.figure(figsize=(20,10))
##pic = imread("猫.png")
#w_c = WordCloud(font_path="C:\Windows\Fonts\simhei.ttf", background_color="white", max_font_size=100, margin=1,width=700,height=600)
#wc = w_c.fit_words({
#    x[0]:x[1] for x in word_count.head(100).values
#})
#plt.imshow(wc, interpolation='bilinear')
#plt.axis("off")
#plt.show()
#
#wc.to_file(titlename+'.png')


##不同关键字对应的售卖情况之和分析
import numpy as np
# 重新更新索引，之前去重的时候没有更新数据data的索引，导致部分行缺失值
data = data.reset_index(drop = True)
#********************************************************************************************
# 不同关键词word对应的数量之和的统计分析
w_s_sum = []
for w in word_count['word']:   
    s_list = []
    for t in title_clean_dist:       
        if w in t:
            s_list.append(int(data.sales[title_clean_dist.index(t)]))                      
    w_s_sum.append(sum(s_list))
#    print('完成一个')
df_w_s_sum = pd.DataFrame({'w_s_sum':w_s_sum})
#print(df_w_s_sum)
# 把 word_count 与对应的 df_w_s_sum 合并为一个表:   
df_word_sum = pd.concat([word_count,df_w_s_sum],axis=1,ignore_index = True)
df_word_sum.columns = ['word','count','w_s_sum']
#print(df_word_sum)
df_word_sum.sort_values('w_s_sum', inplace=True, ascending=True) # 升序
df_w_s = df_word_sum.tail(50)  # 取最大的30行数据
#print(df_w_s)
#******************************************************************************！
#两个字


#******************************************************************************！

import matplotlib
from matplotlib import pyplot as plt
font = {'family' : 'SimHei'}  # 设置字体
matplotlib.rc('font', **font)

def guanjianzi():
    L= []
    if titlename=='成都周边三日':
        L = ['都江堰','青城山','乐山','熊猫','峨眉山','三桥','松坪','武隆'
             '黄龙溪','松潘','色达','长坪','九寨沟','镰刀','黄龙','草原']
    if titlename == '成都周边一日':
        L= ['都江堰','青城山','大佛','熊猫','峨眉山','川剧','川菜',
             '黄龙溪','锦里','武侯祠','快宽窄']
    if titlename == '成都周边二日':
        L= ['都江堰','青城山','乐山','熊猫','峨眉山',
             '黄龙溪','四姑娘山','毕棚','双桥沟','长坪']
    if titlename == '芭蕾舞服':
        L= ['练功','舞蹈','服','芭蕾','芭蕾舞','儿童','形体','成人','体操','短袖','女','女童','裙','吊带','吊带','三沙','连体']
    if titlename == '青海':
        L=['青海湖','茶卡盐湖','张掖','敦煌','莫高窟','祁连','丹霞','塔尔寺','阳关','嘉峪关','西宁']
        
    if titlename == '新疆':
        L= ['喀纳斯','天池','魔鬼城','五彩','吐鲁番','巴音','赛里木湖','那拉提','喀伊','伊犁','禾']
        
    if titlename == '西藏':
        L = ['林芝','羊湖','大峡谷','拉萨','珠峰','纳木错','巴松措','鲁朗','日喀则','冰川','羊卓雍措','布达拉宫','西宁']
         
    import itertools    
    yi =list(itertools.combinations(L,1))  
    yi1 = pd.DataFrame({'yi':yi})          
    yi_list_s = []
    yiavg = []
    for l in yi:        
        yi_list = []
        avglist = []
        for x in title_clean_dist:
            if l[0] in x:
                yi_list.append(int(data.sales[title_clean_dist.index(x)])) 
                avglist.append(int(data.price[title_clean_dist.index(x)]))
                
        yi_list_s.append(sum(yi_list))
        if len(avglist) != 0:
            yiavg.append(sum(avglist)/len(avglist))
        else:
            yiavg.append(sum(avglist))        
    yiavg1 = pd.DataFrame({'number':yiavg})     
    df_yiavg1 = pd.concat([yi1,yiavg1],axis=1,ignore_index = True)
    df_yiavg1.columns = ['liang','number'] 
    #平均价格
#    print(df_yiavg1.values.tolist())        
        
        
    yi_list_s1 = pd.DataFrame({'number':yi_list_s})
    df_yi_list = pd.concat([yi1,yi_list_s1],axis=1,ignore_index = True)
    df_yi_list.columns = ['liang','number'] 
    df_yi_list.sort_values('number',inplace = True,ascending = True) 
    import time
       
    #******************************************************************************！
    #两个字    
    begin = time.time()
    liang =list(itertools.combinations(L,2))  
    liang1 = pd.DataFrame({'liang':liang})          
    #print(liang1)
    liang_list_s = []
    liangavg = []
    for l in liang:
        liang_list = []
        avglist = []        
        for x in title_clean_dist:
            if l[0] in x and l[1] in x:
                liang_list.append(int(data.sales[title_clean_dist.index(x)]))
                avglist.append(int(data.price[title_clean_dist.index(x)]))                
        liang_list_s.append(sum(liang_list))
        if len(avglist) != 0:
            liangavg.append(sum(avglist)/len(avglist))
        else:
            liangavg.append(sum(avglist))
    #    print(l,sum(liang_list))
    liangavg1 = pd.DataFrame({'number':liangavg})
    df_liangavg1 = pd.concat([liang1,liangavg1],axis=1,ignore_index = True)
    df_liangavg1.columns = ['liang','number'] 

#    print(df_liangavg1.values.tolist())

    liang_list_s1 = pd.DataFrame({'number':liang_list_s})    
    df_liang_list = pd.concat([liang1,liang_list_s1],axis=1,ignore_index = True)
    df_liang_list.columns = ['liang','number'] 
    df_liang_list.sort_values('number',inplace = True,ascending = True) 
    df_liang_list= df_liang_list.tail(40)
    end = time.time()
    print('两个字执行时间%.2f'%(end-begin))
    #print(df_liang_list)
       
    #******************************************************************************！
    #三个字
    begin = time.time()
    import itertools
    san = list(itertools.combinations(L,3))  
    san1 = pd.DataFrame({'san':san}) 
    san_list_s = []
    sanavg = []
    for x in san:
        san_list = []
        for y in title_clean_dist:
            if x[0] in y and x[1] in y and x[2] in y: 
                san_list.append(int(data.sales[title_clean_dist.index(y)]))
                avglist.append(int(data.price[title_clean_dist.index(y)])) 
        san_list_s.append(sum(san_list))
        if len(avglist) != 0:
            sanavg.append(sum(avglist)/len(avglist))
        else:
            sanavg.append(sum(avglist))        
    san_list_s = pd.DataFrame({'number':san_list_s})            
    df_san_list = pd.concat([san1,san_list_s],axis=1,ignore_index = True)
    df_san_list.columns = ['san','number']
    df_san_list.sort_values('number',inplace = True,ascending = True)
    df_san_list = df_san_list.tail(40) 
    
    
    sanavg1 = pd.DataFrame({'number':sanavg})
    df_sanavg1 = pd.concat([san1,sanavg1],axis=1,ignore_index = True)
    df_sanavg1.columns = ['liang','number'] 
#    print(df_sanavg1.values.tolist())    
    #print(df_san_list)
    end = time.time()
    print('三个字执行时间%.2f'%(end-begin))    
    #******************************************************************************！
    #四个字
    begin = time.time()
    si= list(itertools.combinations(L,4)) 
    si1 = pd.DataFrame({'si':si}) 
    si_list_s = []
    for x in si:
        si_list = []
        for y in title_clean_dist:
            if x[0] in y and x[1] in y and x[2] in y and x[3] in y: 
                si_list.append(int(data.sales[title_clean_dist.index(y)]))
        si_list_s.append(sum(si_list))
    si_list_s = pd.DataFrame({'number':si_list_s})            
    df_si_list = pd.concat([si1,si_list_s],axis=1,ignore_index = True)
    df_si_list.columns = ['si','number']
    df_si_list.sort_values('number',inplace = True,ascending = True)
    df_si_list = df_si_list.tail(40) 
    end = time.time()
    print('四个字执行时间%.2f'%(end-begin))    
    #******************************************************************************！
    #五个字
    begin = time.time()
    wu= list(itertools.combinations(L,5)) 
    wu1 = pd.DataFrame({'wu':wu}) 
    wu_list_s = []
    for x in wu:
        wu_list = []
        for y in title_clean_dist:
            if x[0] in y and x[1] in y and x[2] in y and x[3] in y and x[4] in y: 
                wu_list.append(int(data.sales[title_clean_dist.index(y)]))
        wu_list_s.append(sum(wu_list))
    wu_list_s = pd.DataFrame({'number':wu_list_s})            
    df_wu_list = pd.concat([wu1,wu_list_s],axis=1,ignore_index = True)
    df_wu_list.columns = ['wu','number']
    df_wu_list.sort_values('number',inplace = True,ascending = True)
    df_wu_list = df_wu_list.tail(40) 
    end = time.time()
    print('五个字执行时间%.2f'%(end-begin))    
    #******************************************************************************！
    #六个字
    from threading import Thread
    from multiprocessing import Queue
    from multiprocessing.pool import ThreadPool
    
    begin = time.time()
    liu= list(itertools.combinations(L,6))     
    liu1 = pd.DataFrame({'liu':liu}) 
    
    liu_list_s = []
    def liuth(x):
        liu_list = []
        for y in title_clean_dist:
            if x[0] in y and x[1] in y and x[2] in y and x[3] in y and x[4] in y and x[5] in y: 
                liu_list.append(int(data.sales[title_clean_dist.index(y)]))
        liu_list_s.append(sum(liu_list))
    pool = ThreadPool(processes=20)
    pool.map(liuth,(x for x in liu))    
    pool.close()   
        
        
    liu_list_s = pd.DataFrame({'number':liu_list_s})            
    df_liu_list = pd.concat([liu1,liu_list_s],axis=1,ignore_index = True)
    df_liu_list.columns = ['liu','number']
    df_liu_list.sort_values('number',inplace = True,ascending = True)
    df_liu_list = df_liu_list.tail(40) 
    end = time.time()
    print('六个字执行时间%.2f'%(end-begin))    
    #******************************************************************************！
    #七个字
#    begin = time.time()
#    qi= list(itertools.combinations(L,7)) 
#    qi1 = pd.DataFrame({'qi':qi}) 
#    qi_list_s = []
#    def qith(x):
#        qi_list = []
#        for y in title_clean_dist:
#            if x[0] in y and x[1] in y and x[2] in y and x[3] in y and x[4] in y and x[5] in y and x[6] in y: 
#                qi_list.append(int(data.sales[title_clean_dist.index(y)]))
#        qi_list_s.append(sum(qi_list))
#    pool = ThreadPool(processes=20)
#    pool.map(qith,(x for x in qi))    
#    pool.close()   
#    qi_list_s = pd.DataFrame({'number':qi_list_s})            
#    df_qi_list = pd.concat([qi1,qi_list_s],axis=1,ignore_index = True)
#    df_qi_list.columns = ['qi','number']
#    df_qi_list.sort_values('number',inplace = True,ascending = True)
#    df_qi_list = df_qi_list.tail(40) 
#    
#    end = time.time()
#    print('七个字执行时间%.2f'%(end-begin))
#    plt.figure(figsize=(10,20))
#    #********************************************************************************************
#    #!关键字对应销量(一个词语)
#    index = np.arange(df_w_s.word.size)
#    plt.barh(index, df_w_s.w_s_sum, color='r', align='center', alpha=0.8)
#    plt.yticks(index, df_w_s.word, fontsize=15)
#    #添加数据标签
#    
#    for y, x in zip(index, df_w_s.w_s_sum):
#        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)
#    plt.savefig(titlename+'关键字对应销量.png')
#    plt.show()
    #一个地点
    plt.figure(figsize=(10,20))
    index=[]
    for x in range(len(yi)):
        index.append(x)
    plt.barh(index, df_yi_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_yi_list.liang, fontsize=15)
    #添加数据标签    
    for y, x in zip(index, df_yi_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)
    plt.savefig(titlename+'一个关键字对应销量.png')
    plt.show()       
    
#    两个关键字对应销量
    plt.figure(figsize=(10,20))
    index=np.arange(df_liang_list.liang.size)
    plt.barh(index, df_liang_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_liang_list.liang, fontsize=15)
    #添加数据标签    
    for y, x in zip(index, df_liang_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)
    plt.savefig(titlename+'两个关键字对应销量.png')
    plt.show()   
    
    #三个字对应销量
    plt.figure(figsize=(10,20))
    index = np.arange(df_san_list.san.size)
    plt.barh(index,df_san_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_san_list.san, fontsize=15)    
    for y, x in zip(index, df_san_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)    
    plt.show() 
#    四个字对应销量
    plt.figure(figsize=(10,20))
    index = np.arange(df_si_list.si.size)
    plt.barh(index,df_si_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_si_list.si, fontsize=15)    
    for y, x in zip(index, df_si_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)    
    plt.show()

#五个字对应销量    
    plt.figure(figsize=(10,20))
    index = np.arange(df_wu_list.wu.size)
    plt.barh(index,df_wu_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_wu_list.wu, fontsize=15)    
    for y, x in zip(index, df_wu_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)    
    plt.show()
    
#六个字对应销量    
    plt.figure(figsize=(10,20))
    index = np.arange(df_liu_list.liu.size)
    plt.barh(index,df_liu_list.number, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_liu_list.liu, fontsize=15)    
    for y, x in zip(index, df_liu_list.number):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)    
    plt.show()   
#七个字对应销量    
#    plt.figure(figsize=(10,20))
#    index = np.arange(df_qi_list.qi.size)
#    plt.barh(index,df_qi_list.number, color='r', align='center', alpha=0.8)
#    plt.yticks(index, df_qi_list.qi, fontsize=15)    
#    for y, x in zip(index, df_qi_list.number):
#        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)    
#    plt.show()    
#guanjianzi()  
def ceshi():   
    plt.figure(figsize=(10,20))
    #********************************************************************************************
    #!关键字对应销量(一个词语)
    index = np.arange(df_w_s.word.size)
    plt.barh(index, df_w_s.w_s_sum, color='r', align='center', alpha=0.8)
    plt.yticks(index, df_w_s.word, fontsize=15)
    #添加数据标签
    
    for y, x in zip(index, df_w_s.w_s_sum):
        plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)
    plt.savefig(titlename+'关键字对应销售额.png')
    plt.show()
    #********************************************************************************************
    #plt.barh(index, df_gjzs.w_s_sum, color='blue', align='center', alpha=0.8)
    #plt.yticks(index, df_gjzs.word, fontsize=15)
    ##添加数据标签
    #for y, x in zip(index, df_gjzs.w_s_sum):
    #    plt.text(x, y, '%.0f' %x , ha='left', va='center', fontsize=15)
    #plt.show()
    #********************************************************************************************
    #价格分布情况分析：
    plt.figure(figsize=(7,5))
    plt.hist(data['price'], bins=15, color='g')
    plt.xlabel('价格', fontsize=25)
    plt.ylabel('商品数量', fontsize=25)
    plt.title(titlename+'不同价格对应的商品数量分布', fontsize=17)
    plt.savefig(titlename+'价格分布情况.png')
    plt.show()
    
    #plt.savefig('价格分布.png')
    #********************************************************************************************
    #销量分布
    data_s = data[data['sales'] >data.sales.mean()]
    
    #print('销量500以上的商品占比：%.3f'%(len(data_s) / len(data)))
    
    plt.figure(figsize=(12,8))
    plt.hist(data_s['price'],bins=20, color='y')   # 分二十组
    plt.xlabel('价格区间', fontsize=25)
    plt.ylabel('销量', fontsize=25)
    
    plt.title(titlename+'销售额大于平均值以上的商品价格分布', fontsize=25)
    plt.savefig(titlename+'销售额大于平均值分布.png')
    plt.show()
    
    #********************************************************************************************
    #商品对销量的影响
#    flg, ax = plt.subplots()
#    ax.scatter(data['price'],
#              data['sales'], color='blue')
#    ax.set_xlabel('价格')
#    ax.set_ylabel('销售额')s
#    ax.set_title('商品价格对销量的影响')
#    plt.savefig(titlename+'价商品对销售额的影响.png')
#    plt.show()
    
    #********************************************************************************************
    #商品价格对销售额的影响分析
    print('商品价格对销售额的影响线性回归图')
    data['GMV'] = data['sales'] 
    import seaborn as sns
    sns.regplot(x='price', y='GMV', data=data, color='b')
    #s.savefig('商成都线性回归图.png')
    print('价格对销售额的影响系数：%.7f'%(data.price.
                              corr(data.sales)))

    
    
    #********************************************************************************************
    #不同时间的销售额分布
#    
#    plt.figure(figsize=(12,8))
#    data.time.value_counts().plot(
#                    kind='bar', color='purple')
#    plt.xticks(rotation=0)        
#    plt.xlabel('时间')
#    plt.ylabel('数量')
#    plt.title(titlename+'不同时间的商品数量分布')
#    plt.savefig(titlename+'时间分布.png')
#    plt.show()
#   



#ceshi()
print('是否进行初步分析')
s = input('请输入:')

if s == '是':
    ceshi()

else:
    pass
print('是否进行标题多维度分析：')
s1 = input('请输入：')
if s1 == '是':
    guanjianzi()

else:
    pass    
















#分词
def jieba1():
    str = ''
    for x in range(378):
        str += data['title'][x]
#    print(str)
    
    seg = jieba.cut(str,cut_all=False)
    str1 = ' '.join(seg).split()
    c = Counter()
    
    for x in str1:
        if len(x) > 1 and x != '\r\n':
            c[x] += 1
    
    for (k,v) in c.most_common(100):# 输出词频最高的前两个词
        print("%s:%d"%(k,v))
#    keywords = jieba.analyse.extract_tags(str, topK=20, withWeight=True, allowPOS=())
#    for item in keywords:
#    # 分别为关键词和相应的权重
#        print(item[0], item[1])
    
#jieba1()
#data = datatmsp[['title','money','people']]
#print(data.head())

#print(title)
#stopwords = stopwords.stopword.values.tolist()
#print(stopwords)