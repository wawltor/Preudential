'''
Created on 2015/12/27

@author: FZY
'''
import pandas as pd 
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import random
#59381
if __name__ == '__main__':
    #in this model , I will use count the family feature 
    train = pd.read_csv("../../data/train_clean_21.csv")
    test = pd.read_csv("../../data/test_clean_21.csv")
    train['FH_24_1'] = train['Family_Hist_2']*train['Family_Hist_4']
    train['FH_24_2'] = train['Family_Hist_2']+train['Family_Hist_4']
    train['FH_24_3'] = np.abs(train['Family_Hist_2']-train['Family_Hist_4'])
    train['FH_24_4'] = train['Family_Hist_2']/train['Family_Hist_4']
    train['FH_24_5'] = train['Family_Hist_4']/train['Family_Hist_2']
    train['FH_all_1'] = train['Family_Hist_1']+train['Family_Hist_2']+train['Family_Hist_3']+train['Family_Hist_4']+train['Family_Hist_5']
    train['FH_all_2'] = train['Family_Hist_1']*train['Family_Hist_2']*train['Family_Hist_3']*train['Family_Hist_4']*train['Family_Hist_5']
    test['FH_24_1'] = test['Family_Hist_2']*test['Family_Hist_4']
    test['FH_24_2'] = test['Family_Hist_2']+test['Family_Hist_4']
    test['FH_24_3'] = np.abs(test['Family_Hist_2']-test['Family_Hist_4'])
    test['FH_24_4'] = test['Family_Hist_2']/test['Family_Hist_4']
    test['FH_24_5'] = test['Family_Hist_4']/test['Family_Hist_2']
    test['FH_all_1'] = test['Family_Hist_1']+test['Family_Hist_2']+test['Family_Hist_3']+test['Family_Hist_4']+test['Family_Hist_5']
    test['FH_all_2'] = test['Family_Hist_1']*test['Family_Hist_2']*test['Family_Hist_3']*test['Family_Hist_4']*test['Family_Hist_5']
    train['FH_24_4'][train['Family_Hist_4']==0.0] = 0
    test['FH_24_4'][test['Family_Hist_4']==0.0] = 0
    train['FH_24_5'][train['Family_Hist_2']==0.0] = 0
    test['FH_24_5'][test['Family_Hist_2']==0.0] = 0
    train.to_csv("../../data/train_clean_21.csv",index=False)
    test.to_csv("../../data/test_clean_21.csv",index=False)

    
    