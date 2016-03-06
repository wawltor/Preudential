'''
Created on 2016/1/12

@author: FZY
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from ml_metrics import *
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from utils import *
param_space_reg_skl_etr = {
    'task': 'reg_skl_etr',
    'n_estimators': 300,
    'max_features': 0.28,
    'n_jobs': 2,
    'random_state': 2015,
    "max_evals": 1,
}

param_space_reg_skl_gbm = {
    'task': 'reg_skl_gbm',
    'n_estimators': 250,
    'learning_rate': 0.04,
    'max_features': 0.7,
    'max_depth': 8,
    'max_samples_leaf':4,
    'subsample': 0.7,
    'random_state': 2015,
    "max_evals": 1,
}



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

    train = pd.read_csv("../../data/train_clean_22.csv")
    test = pd.read_csv("../../data/test_clean_22.csv") 
    cps = np.zeros([3,7])
    params = [270,350,260]
    for run in range(2,3):
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
        cols = list(test.columns)
        cols.remove('Id')
        train_features = cols
        X_train = train.iloc[train_index][train_features]
        X_valid = train.iloc[test_index][train_features]
        labels_train = train.iloc[train_index]['Response']
        labels_valid = train.iloc[test_index]['Response']
        #first use etc_enc to predict the data
        param = param_space_reg_skl_etr
        etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
        etr.fit(X_train,labels_train)
        pred = etr.predict(X_valid)
        y_train_preds = etr.predict(X_train)
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,labels_valid)
        print kappa
        train_sort = pd.DataFrame({'cols':train_features,'value':list(etr.feature_importances_)}).sort(columns=['value'],ascending=False)
        train_sort.to_csv("sort.csv",index=False)
    
    
    
    