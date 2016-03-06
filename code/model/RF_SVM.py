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
from ftrl import *
from ftrl_softmax import *
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from ftrl_Regression import  FtrlRegressor
from sklearn.learning_curve import validation_curve
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
param_space_reg_skl_etr = {
    'task': 'reg_skl_etr',
    'n_estimators': 300,
    'max_features': 0.28,
    'n_jobs': 32,
    'random_state': 2015,
    "max_evals": 1,
}

param_space_reg_skl_gbm = {
    'task': 'reg_skl_gbm',
    'n_estimators': 270,
    'learning_rate': 0.04,
    'max_features': 0.7,
    'max_depth': 8,
    'max_samples_leaf':4,
    'subsample': 0.7,
    'random_state': 2015,
    "max_evals": 1,
}
param_space_reg_skl_svr = {
    'task': 'reg_skl_svr',
    'C': 2,
    'epsilon':0.1,    
    'kernel':  'rbf',
    "max_evals": 1,
    
}
def rmse(X,y):
        res = np.sqrt(mean_squared_error(X,y))
        return res

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
    sort_features = pd.read_csv("sort.csv")
    train_features = list(sort_features.iloc[0:200]['cols'].values)
    param = param_space_reg_skl_svr
    for run in range(2,3):
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
        X_train = train.iloc[train_index][train_features]
        X_valid = train.iloc[test_index][train_features]
        labels_train = train.iloc[train_index]['Response'] 
        labels_valid = train.iloc[test_index]['Response'] 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        svr = SVR(C=param['C'], epsilon=param['epsilon'],cache_size=1024, kernel=param['kernel'],verbose=True)
        svr.fit(X_train, labels_train)
        print 'training end!'
        pred = svr.predict(X_valid)
        Y_train = svr.predict(X_train)
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,labels_valid) 
        print kappa
    imput = Imputer()
    X_test = test[train_features]
    X_test = imput.fit(X_test)
    Y_test = svr.predict(X_test)
    print 'predict end!'
    
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    y_pred = pd.cut(Y_test,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    id_test = test['Id']
    out_put = pd.DataFrame({'id':id_test,'Response':y_pred})
    out_put.to_csv("../../result/svr.csv",index=False) 
   
    
        

  
    
        
    
   
    
    
    
    
   
    
    
    
    
    