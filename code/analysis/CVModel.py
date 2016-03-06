'''
Created on 2015/11/10

@author: FZY
'''

#for creat the same the data in CV, we must the use the cv
#in this model, I will create the CV  for create ++
import pandas as pd
from sklearn.cross_validation import train_test_split
def creatCVIndex():
    train = pd.read_csv("../../data/train.csv")
    X_train,X_test,Y_train,Y_test=train_test_split(train,train.index,test_size=0.2,random_state=1)
    dumpCVIndex("../../data/cv/train.run1.txt", list(Y_train))
    dumpCVIndex("../../data/cv/valid.run1.txt", list(Y_test))
    X_train,X_test,Y_train,Y_test=train_test_split(train,train.index,test_size=0.2,random_state=2)
    dumpCVIndex("../../data/cv/train.run2.txt", list(Y_train))
    dumpCVIndex("../../data/cv/valid.run2.txt", list(Y_test))
    X_train,X_test,Y_train,Y_test=train_test_split(train,train.index,test_size=0.2,random_state=3)
    dumpCVIndex("../../data/cv/train.run3.txt", list(Y_train))
    dumpCVIndex("../../data/cv/valid.run3.txt", list(Y_test))

def dumpCVIndex(path,list):
    ind = 1
    f = open(path,"wb")
    for i in list:
        if ind == 1:
            f.write(str(i))
        else:
            f.write(",%d"%(i))
        ind = ind + 1
    f.close()

def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist

if __name__ == '__main__':
    
    creatCVIndex()

    