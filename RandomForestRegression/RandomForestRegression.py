import csv
import random
import time
import itertools
import math
import msvcrt as m

min_nr_sample=50 #minimum % number of items in a sample
max_nr_sample=100 #minimum % number of items in a sample
nrtree=10 #number of trees fused in the random forest
nrsplit=10 #number of splits for comparable variables
onlyhour=True
def read(file):
    '''
    read the data and return it as a list
    Parameters:
    filename
    '''
    with open(file) as f:
        reader = csv.reader(f)
        data = [r for r in reader]
    return data[0],data[1:]
def makefloat(data):
    '''
    change str to float
    '''
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            if j == 0:
                astr=data[i][j]
                astr=astr.split(' ')
                astr[0]=astr[0].split('-')
                astr[1]=astr[1].split(':')
                if not onlyhour:
                    astr=astr[0]+astr[1]
                else:
                    astr=astr[1]
                sum=0
                for k in astr:
                    sum*=100
                    sum+=float(k)
                data[i][j]=sum
            else:
                data[i][j]=float(data[i][j])


    return None
def dropcolumn(data,index):
    '''
    Drops a specific column
    '''
    for row in data:
        row.pop(index)
def sample(data,min_nr_sample=50,max_nr_sample=100):
    '''
    sample with replacement of the data
    '''  
    return random.sample(data, random.randint(int(len(data)/100*min_nr_sample), int(len(data)/100*max_nr_sample))) 
def split(data, func):
    '''
    splits the tree into two parts by the function
    '''
    left,right=[],[]
    for i in data:
        if func(i):
            right.append(i)
        else:
            left.append(i)
    return left,right
def error(data):
    avg = 0.0
    for i in data:
        avg+=i[-1]
    avg/= len(data)

    error = 0.0
    for i in data:
        error+= (avg - i[-1])**2
    error/= len(data)
    return error
def buildtree(data,o_criteria,a_criteria,prev_avg=-1):
    '''
    build a tree with the given subspace 
    '''
    if len(data)==0:
        return [prev_avg]
    avg = 0.0
    for i in data:
        avg+=i[-1]
    avg/= len(data)
    if len(data)<2:
        return [avg]
    criteria=sample(a_criteria, 75 ,100)
    if criteria==[]:
        return [avg]
    best_c,best_v,best_e=["",False],9999,float("inf")
    for c in criteria:
        ci=index(o_criteria,c)
        sdata=sorted(data,key=lambda x:x[ci])
        if(c[1]):
            for spliti in range(0,nrsplit-1):

                ind=int(len(data)/nrsplit*(spliti+1))
                l,r=split(sdata,lambda x: x[ci] >= data[ind][ci])
                if len(l) == 0:
                    terror=error(r)
                elif len(r) == 0:
                    terror=error(l)
                else:
                    terror=error(l)
                    terror+=error(r)
                if(terror<best_e or best_e==float("inf")) and len(l) != 0 and len(r) !=0 :
                    best_c=c
                    best_v=data[ind][ci]
                    best_e=terror
        else:
            tempi=[x[ci] for x in data]
            tempset=set(tempi)
            tempi=list(tempset)

            for cat in tempi:
                l,r=split(sdata,lambda x: x[ci]==cat)
                if len(l) == 0:
                    terror=error(r)
                elif len(r) == 0:
                    terror=error(l)
                else:
                    terror=error(l)
                    terror+=error(r)
                if(terror<best_e or best_e==float("inf")):
                    best_c=c
                    best_v=cat
                    best_e=terror


    l,r=split(data,lambda x: x[index(o_criteria,best_c)]>=best_v)
    cu=list(a_criteria)
    #if len(l) == 0 or len(r) == 0:
    #    cu.remove(best_c)
    #print(str(len(l))+", "+str(len(r))+"    "+best_c[0]+"    "+str(best_v))
    if best_c[0]=="":
        return [avg]
    if len(l)==0:
        return buildtree(r,o_criteria,cu,avg)
    if len(r)==0:
        return buildtree(l,o_criteria,cu,avg)
    if(c[1]):
        return [lambda x: x[index(o_criteria,best_c)]>=best_v, buildtree(l,o_criteria,cu,avg),buildtree(r,o_criteria,cu,avg)]
    else:
        return [lambda x: x[index(o_criteria,best_c)]==best_v, buildtree(l,o_criteria,cu,avg),buildtree(r,o_criteria,cu,avg)]

def prediction(trees, item):
    '''
    returns the prediction
    '''
    avg=0.0
    for t in trees:
        avg+=pred_val(t,item)
    avg/=len(trees)

    return avg
def prediction_int(trees, item):
    '''
    returns the prediction
    '''
    avg=0.0
    for t in trees:
        avg+=pred_val(t,item)
    avg/=len(trees)

    return int(round(avg))

def pred_val(tree, item):
    if len(tree)==1:
        return tree[0]
    if tree[0](item):
        return pred_val(tree[2],item)
    else:
        return pred_val(tree[1],item)

def index(l,item):
    for i in range(0,len(l)):
        if l[i]==item:
            return i
    return -1
def fin_error(data,tree):
    error=0.0
    for i in data:
        error+=(i[-1]-pred_val(tree,i))**2
    error/=len(data)
    return error
def fin_error2(data,trees,prediction):
    error=0.0
    for i in data:
        p=prediction(trees,i)
        error+=(i[-1]-p)**2
    error/=len(data)
    return error
def biggest_dif(data,tree):
    val,pred=0.0,0.0
    for i in data:
        if abs((i[-1]-pred_val(tree,i)))>abs(val-pred):
            val=i[-1]
            pred=pred_val(tree,i)
    return val,pred
def biggest_dif2(data,trees,prediction):
    val,pred=0.0,0.0
    for i in data:
        pr=prediction(trees,i)
        if abs((i[-1]-pr))>abs(val-pred):
            val=i[-1]
            pred=pr
    return val,pred
def smallest_dif2(data,trees,prediction):
    val,pred=0.0,100000000000000000000.0
    for i in data:
        pr=prediction(trees,i)
        if abs((i[-1]-pr))<abs(val-pred):
            val=i[-1]
            pred=pr
    return val,pred

seed=0 #time.time()

random.seed(seed)
h,d=read("train.csv")
dropcolumn(d,9)
dropcolumn(d,9)
h.pop(9)
h.pop(9)
makefloat(d)
#comparable=[True,False,False,False,True,True,True,True,True]
comparable =[True,True,True,True,True,True,True,True,True]
h = list(map(lambda x,y: [x,y] , h , comparable))
print(error(d))
trees=[]
for i in range(0,nrtree):
    print(i)
    ds=sample(d)
    tt=buildtree(ds,h,h)
    trees.append(tt)

    p=prediction
print(fin_error2(d,trees,p))
val,pred=biggest_dif2(d,trees,p)
print("biggest difference:\nval:"+str(val)+" pred:"+str(pred))
val,pred=smallest_dif2(d,trees,p)
print("smallest difference:\nval:"+str(val)+" pred:"+str(pred))

import numpy as np 
import matplotlib.pyplot as plt

oval=[]
predval=[]
for e in d:
    oval.append(e[-1])
    predval.append(p(trees,e))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(np.array(range(1,len(d)+1)),np.array(oval),c='y', label="Original")
ax.plot(np.array(range(1,len(d)+1)),np.array(predval),c='r', label="Predicted")
ax.set_xlabel("Depth")
ax.set_ylabel("Error%")
ax.legend(loc='upper right')
fig.show()

'''
import pandas as pd  

dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,11].values

for e in range(0,len(d)):
    astr= X[e,0]
    astr=astr.split(' ')
    astr[0]=astr[0].split('-')
    astr[1]=astr[1].split(':')
    if not onlyhour:
        astr=astr[0]+astr[1]
    else:
        astr=astr[1]
    sum=0
    for k in astr:
        sum*=100
        sum+=float(k)
    X[e,0]=sum

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)  
regressor.fit(X, y)  
y_pred = regressor.predict(X)  
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred))) 

oval=[]
predval=[]
for e in d:
    oval.append(e[-1])
    predval.append(p(trees,e))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(np.array(range(1,len(d)+1)),np.array(oval),c='y', label="Original")
ax.plot(np.array(range(1,len(d)+1)),y_pred,c='r', label="Predicted")
ax.set_xlabel("Depth")
ax.set_ylabel("Error%")
ax.legend(loc='upper right')
fig.show()
'''