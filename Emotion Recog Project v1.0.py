#!/usr/bin/env python
# coding: utf-8

# In[80]:


import sys
import pandas as pd
import os
import numpy as np
import math
from math import isnan
import copy

import mcdm

import ahpy
from decipy import executors as exe
from pymcdm.methods import TOPSIS, VIKOR
from pymcdm.helpers import rrankdata
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
from pymcdm import weights as mcdm_weights

sys.path.insert(0, 'F:\Vishu\BITS\Image Analysis Project')
import pyds
from pyds import MassFunction
from itertools import product
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"F:\Vishu\BITS\Emotion Recog Project\train.csv")


# In[2]:


#removing all data containing the neutral class
data = data.loc[data["Label"]!=1]
# print(data)


# In[56]:


train_ratio = int(0.8*len(data))
train_X = data.iloc[:train_ratio,:]
train_X = np.array(train_X)
train_X = train_X[train_X[:, -1].argsort()]
train_class = train_X[:, -1]
train = train_X[:, :-1]

pd.DataFrame(train).to_csv(r"F:\Vishu\BITS\Emotion Recog Project\X_train.csv")
pd.DataFrame(train_class).to_csv(r"F:\Vishu\BITS\Emotion Recog Project\Y_train.csv")

test_X = data.iloc[train_ratio:,:]
test_X = np.array(test_X)
test_class = test_X[:,-1]
test_class = test_class.astype(int)
test = test_X[:,:-1]

pd.DataFrame(test).to_csv(r"F:\Vishu\BITS\Emotion Recog Project\X_test.csv")
pd.DataFrame(test_class).to_csv(r"F:\Vishu\BITS\Emotion Recog Project\Y_test.csv")

#do 80-20 split 
#make bpa file out of test 

listhead = list(data)
dataArr = data.to_numpy()
label = dataArr[:, -1]
#print(test)
# print(train_class)


# In[57]:


cls = [2, 3, 4, 5, 6, 7, 8]
pcs = list(range(1, len(test)+1))


# In[58]:


# ans = [[0 for i in range(len(cls)+1)] for j in range(len(pcs)+1)]
# ans[0][0] = 0

# for i in range(1,len(ans[0])):
#     ans[0][i]=cls[i-1]
    
# for i in range(1, len(ans)):
#     ans[i][0]=pcs[i-1]


# In[59]:


row, row2, cols = 360, 7, 3
triArr = np.array([[[0.0]*cols]*row2]*row)
for k in range(360):
    j = 0
    for i in range(1,8):
        cls_feature = []
        while(j<len(train)):            
            if(i+1==train_class[j]):
                cls_feature.append(train[j][k])
                j+=1
            else:
                break
        cls_feature = np.array(cls_feature)
        if(i+1>1):
            triArr[k][i-1][0] = min(cls_feature)
            triArr[k][i-1][1] = np.mean(cls_feature)
            triArr[k][i-1][2] = max(cls_feature)         
#input triar for overlap


# In[60]:


def membership(x, ls):
    #ls min, mean,. max
    if(x<=ls[0] or x>=ls[2]):
        return 0
    if(x>=ls[1]):
        return (ls[2]-x)/(ls[2]-ls[1])
    else: 
        return (x-ls[0])/(ls[1]-ls[0])


# In[61]:


#overlap area calcc
#for each attribute create a 7x7 matrix 
#overall matrix be 360*7*7                                            min      mean     max
#in a graph for triangular membership function your coordinates are (c1, 0); (c2, 1); (c3, 0)
#so for finding intersection you just use these coordinated to find overlap


# In[62]:


def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0
    
    d = (det(*line1), det(*line2))
    #x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return y


# In[63]:


def area(y, base):
    return 0.5*y*base


# In[64]:


def normalize(arr):
    norm_arr = []
    diff = 0.9
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr)+0.1
        norm_arr.append(temp)
    return norm_arr


# In[65]:


pre_overlap = []
for k in range(len(triArr)):
    #find overlapp for each class with other classes
    temp1 = []
    for j in range(7):
        #pre_overlap[k][j][i]=area(y1, base)
        temp2 = []
        for i in range(7):
            line1, line2 = [[0]*2]*2, [[0]*2]*2
            base = 0
            if(triArr[k][j][1]>triArr[k][i][1]):
                #when c22>c12 you consider A(c22, c21) and B(c12, c13)
                line1[0]=[triArr[k][j][1], 1]         
                line1[1]=[triArr[k][j][0], 0]
                line2[0]=[triArr[k][i][1], 1]         
                line2[1]=[triArr[k][i][2], 0]
                #c13-c21
                base = triArr[k][i][2] - triArr[k][j][0]
            else: 
                #when c22<c12 you consider A(c22, c23) and B(c12, c11)
                line1[0]=[triArr[k][i][1], 1]         
                line1[1]=[triArr[k][i][0], 0]
                line2[0]=[triArr[k][j][1], 1]         
                line2[1]=[triArr[k][j][2], 0]
                #c23-c11
                base = triArr[k][j][2] - triArr[k][i][0]
            y1 = max(0.0, intersection(line1, line2))
            #print(y1, base, area(y1, base))
            temp2.append(area(y1, base))
        #temp2 = normalize(temp2)
        temp1.append(temp2)
    pre_overlap.append(temp1)
#print(pre_overlap)


# In[66]:


overlap = np.array([[0.0]*7]*360)
for j in range(360):
    for i in range(7):
        overlap[j][i]=np.sum(pre_overlap[j][i])-pre_overlap[j][i][i]
        #print(pre_overlap[j])
    #standardize
    #overlap[j] = normalize(overlap[j])
print(overlap)


# In[67]:


pd.DataFrame(overlap).to_csv(r"F:\Vishu\BITS\Emotion Recog Project\overlap.csv")


# In[68]:


print(test.shape)


# In[69]:


print(test_class)


# In[70]:


test_bpa_full=[]

res_mew=[]
res_topsis=[]
res_saw=[]

for k in range(len(test)):
    #test_bpa_single=[[0.0]*7]*360
    test_bpa_single = []
    test_bpa_dict = []
    for j in range(360):
        temp = []
        dict_temp = dict()
        for i in range(7):
            mux = membership(test[k][j], triArr[j][i])
            if(mux==0): mux = 0.0000001
            temp.append(mux/(mux+overlap[j][i]))
            dict_temp[str(i+2)] = mux/(mux+overlap[j][i])
            
        test_bpa_single.append(temp)
        test_bpa_dict.append(dict_temp)
        
    #Dempster-Schafer
    test_bpa_full.append(test_bpa_dict)
    
    #mcdm
    test_bpa_sin_trans = np.transpose(test_bpa_single)
    #MEW
    mew=mcdm.rank(test_bpa_sin_trans, s_method="MEW")
    res_mew.append(int(mew[0][0][1])+1)
    
    #TOPSIS
    topsis=mcdm.rank(test_bpa_sin_trans, s_method="TOPSIS")
    
    res_topsis.append(int(topsis[0][0][1])+1)
    
    #SAW
    saw=mcdm.rank(test_bpa_sin_trans, s_method="SAW")
    res_saw.append(int(saw[0][0][1])+1)
    


# In[71]:


acc_mew=0
acc_topsis = 0
acc_saw = 0
for i in range(len(test_class)):
    if(test_class[i]==res_mew[i]): acc_mew+=1
    if(test_class[i]==res_topsis[i]): acc_topsis+=1
    if(test_class[i]==res_saw[i]): acc_saw+=1

acc_mew/=len(test_class)
acc_topsis/=len(test_class)
acc_saw/=len(test_class)

print("MEW acc= ", acc_mew)

print("TOPSIS acc= ",acc_topsis)

print("SAW acc= ",acc_saw)


# In[72]:


#test_bpa_full=np.array(test_bpa_full)
#print(test_bpa_full)


# In[78]:


test_class = test_class.astype(str)

acc = 0
vals=[]
for i in range(len(test_bpa_full)):
    initial = MassFunction(test_bpa_full[i][0])
    for j in range(1, len(test_bpa_full[i])):
        initial = initial&MassFunction(test_bpa_full[i][j])   
    pred_label = []
    #sort_orders = sorted(initial.items(), key=lambda x: x[1], reverse=True)
    #print(initial)
    pred_label=list(initial.max_pl())
    #print(pred_label)
    vals.append(pred_label[0])
    #print(str(test_class[i]))
    
    if(pred_label[0] == test_class[i]): acc+=1
         
print("Accuracy_hard = ", acc/len(test_bpa_full)) 


# In[81]:


labels = ['2', '3', '4', '5', '6', '7', '8']
print(metrics.classification_report(test_class, vals, labels))
cm = metrics.confusion_matrix(test_class, vals, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




