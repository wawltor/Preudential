'''
Created on 2015/12/23

@author: FZY
'''
import pandas as pd 
import numpy as np
import scipy as sp
import random
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist
#BMI*HT
#BMI*WT
#BMI/H32
#

if __name__ == '__main__':
    #in this section , I will get some message of BMI
    #first add feature BMI * Age
  
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv")
    train['scale'] = 0.0061*train['Ht']+0.0128*train['Wt']
    test['scale'] = 0.0061*test['Ht']+0.0128*test['Wt']
    train['Fat'] = 1.2*train['BMI']+0.23*train['Ins_Age']
    test['Fat'] = 1.2*test['BMI']+0.23*test['Ins_Age']
    train['Ht_BMI'] = train['Ht']*train['BMI']
    train['BMI_H32'] = train['BMI']/train['Medical_History_32']
    test['Ht_BMI'] = test['Ht']*test['BMI']
    test['BMI_H32'] = test['BMI']/test['Medical_History_32']
    train['BMI_Age'] = train['BMI'] * train['Ins_Age']
    test['BMI_Age'] = test['BMI'] * test['Ins_Age']
    train['Wt_Age'] = train['Wt'] - train['Ins_Age']
    train['Wt_Age'] = train['Wt'] - train['Ins_Age']
    train['BMI_H32'][train['Medical_History_32']==0.0] = 0.0
    test['BMI_H32'][test['Medical_History_32']==0.0] = 0.0
    train.to_csv("../../data/train_clean_21.csv",index=False)
    test.to_csv("../../data/test_clean_21.csv",index=False)
    
  
    
   
    
    