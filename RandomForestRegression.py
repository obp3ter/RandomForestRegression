import csv
import random
import time
import itertools
import math

min_nr_sample=100 #minimum % number of items in a sample
nrtree=1 #number of trees fused in the random forest
nrsplit=10 #number of splits for comparable variables

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
                astr=astr[0]+astr[1]
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
def sample(data):
    '''
    sample with replacement of the data
    '''  
    return random.sample(data, random.randint(len(data)*100/min_nr_sample, len(data))) 
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
def buildtree(data,criteria,o_criteria):
    '''
    build a tree with the given subspace 
    '''
    if len(data)==0:
        return [-1.0]
    if criteria==[]:
        avg = 0.0
        for i in data:
            avg+=i[-1]
        avg/= len(data)
        return [avg]

    best_c,best_v,best_e=["",False],9999,float("inf")
    for c in criteria:
        ci=index(o_criteria,c)
        sdata=sorted(data,key=lambda x:x[ci])
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
            if(terror<best_e or best_e==float("inf")):
                best_c=c
                best_v=data[ind][ci]
                best_e=terror
    cu=list(criteria)
    cu.remove(best_c)
    l,r=split(data,lambda x: x[index(o_criteria,best_c)]>=best_v)
    return [lambda x: x[index(o_criteria,best_c)]>=best_v, buildtree(l,cu,o_criteria),buildtree(r,cu,o_criteria)]

def prediction(trees, item):
    '''
    returns the prediction
    '''
    return None

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

random.seed(time.time())
h,d=read("train.csv")
dropcolumn(d,9)
dropcolumn(d,9)
h.pop(9)
h.pop(9)
makefloat(d)
#comparable=[True,False,False,False,True,True,True,True,True]
comparable =[True,True,True,True,True,True,True,True,True]
h = list(map(lambda x,y: [x,y] , h , comparable))
print(float("inf")==float("inf"))
t=buildtree(d,h,h)
print(error(d))
print(fin_error(d,t))