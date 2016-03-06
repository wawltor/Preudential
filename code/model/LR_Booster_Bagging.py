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

def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist

def getScore(round):
    round = round
    grd = GradientBoostingRegressor(
        loss = 'huber',
        learning_rate=0.04,
        n_estimators=round,# 100 is enough using this learning_rate
        max_depth=6,
        subsample=0.7,
        max_features=0.7,
        min_samples_leaf=1,
        verbose=2,
        random_state=2015,
    )
    grd.fit(X_train,labels_train)
    Y = grd.predict(X_valid)
    Y_train = grd.predict(X_train)
    cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
    res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train,labels_train),method='Nelder-Mead')
    cutpoints = np.sort(res.x)
    print cutpoints
    kappa=minimize_quadratic_weighted_kappa(cutpoints,Y,labels_valid)  
    print kappa
    
if __name__ == '__main__':
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv") 
    cps = np.zeros([3,7])
    params = [270,350,260]
    for run in range(1,4):
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
        cols = list(test.columns)
        cols.remove('Id')
        train_features = cols
        X_train = train.iloc[train_index][train_features]
        
        X_valid = train.iloc[test_index][train_features]
        labels_train = train.iloc[train_index]['Response']
        labels_valid = train.iloc[test_index]['Response']
        
        """
        #first use etc_enc to predict the data
        etc_enc = OneHotEncoder()
        etr = ExtraTreesClassifier(n_estimators=param_space_reg_skl_etr['n_estimators'],
                                          max_features=param_space_reg_skl_etr['max_features'],
                                          n_jobs=param_space_reg_skl_etr['n_jobs'],
                                          random_state=param_space_reg_skl_etr['random_state'])
        etc_lm= LogisticRegression()
        etr.fit(X_train,y_train)
        etc_enc.fit(etr.apply(X_train))
        etc_lm.fit(etc_enc.transform(X_train_lr),y_train_lr)
        y_train_lr_pred = etc_lm.predict(etc_enc.transform(etr.apply(X_valid)))
        score = quadratic_weighted_kappa(y_train_lr_pred,labels_valid)
        print score
        """
        """
        rf = RandomForestClassifier(n_estimators=param_space_reg_skl_etr['n_estimators'],
                                          max_features=param_space_reg_skl_etr['max_features'],
                                          n_jobs=param_space_reg_skl_etr['n_jobs'],
                                          random_state=param_space_reg_skl_etr['random_state'])
        rf_enc = OneHotEncoder()
        rf_lm = LogisticRegression()
        rf.fit(X_train, y_train)
        rf_enc.fit(rf.apply(X_train))
        rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
        y_pred_rf_lm = rf_lm.predict(rf_enc.transform(rf.apply(X_valid)))
        score = quadratic_weighted_kappa(y_pred_rf_lm,labels_valid)
        print score
        """
        grd = GradientBoostingRegressor(
            loss = 'huber',
            learning_rate=0.04,
            n_estimators=params[run-1],# 100 is enough using this learning_rate
            max_depth=6,
            subsample=0.7,
            max_features=0.7,
            min_samples_leaf=1,
            verbose=0,
            random_state=2015,
        ) 
        grd_enc = OneHotEncoder()
        result ={}
        grd.fit(X_train,labels_train)
        grd_enc.fit(grd.apply(X_train))
        """
        etr = ExtraTreesRegressor(n_estimators=param_space_reg_skl_etr['n_estimators'],
                                          max_features=param_space_reg_skl_etr['max_features'],
                                          n_jobs=param_space_reg_skl_etr['n_jobs'],
                                          random_state=param_space_reg_skl_etr['random_state'])
        etr.fit(grd_enc.transform(grd.apply(X_train)),labels_train)
        Y_train = etr.predict(grd_enc.transform(grd.apply(X_train)))
        Y = etr.predict(grd_enc.transform(grd.apply(X_valid)))
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        kappa=minimize_quadratic_weighted_kappa(cutpoints,Y,labels_valid)
        """  
        ridge = Ridge(alpha=2500)
        ridge.fit(grd_enc.transform(grd.apply(X_train)),labels_train)
        Y_train = ridge.predict(grd_enc.transform(grd.apply(X_train)))
        Y = ridge.predict(grd_enc.transform(grd.apply(X_valid)))
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        cps[run-1] = cutpoints
        kappa=minimize_quadratic_weighted_kappa(cutpoints,Y,labels_valid) 
        print kappa
    print cps
    cutpoints = list(np.mean(cps,axis=0))    
    X_test = test[train_features]
    Y_test = ridge.predict(grd_enc.transform(grd.apply(X_test)))
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    print cutpoints
    y_pred = pd.cut(Y_test,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    id_test = test['Id']
    out_put = pd.DataFrame({'id':id_test,'Response':y_pred})
    out_put.to_csv("../../result/ridge.csv",index=False) 
  
    
        
    
   
    
    
    
    
   
    
    
    
    
    