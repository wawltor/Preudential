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
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv") 
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
        all_data = pd.concat([X_train,labels_train],axis=1)
        all_data.index = [i for i in range(0,47504)]
        #get gbdt model
        grd = joblib.load("ftrl_run2.model")
        grd_enc = OneHotEncoder()
        grd_enc.fit(grd.apply(X_train))
        ftrl = FtrlRegressor(alpha=0.001, beta=1,nb_epoch=260,batch_size=1024, l1=0.1, l2=0.5,early_stop_rounds=False,eval_function=rmse)  
        ftrl.fit(grd_enc.transform(grd.apply(all_data[train_features])),all_data['Response'],validation_set=True,validation_split = 0.2)
        Y = ftrl.predict(grd_enc.transform(grd.apply(X_valid)))
        Y_train = ftrl.predict(grd_enc.transform(grd.apply(all_data[train_features])))
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train,all_data['Response']),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        cps[run-1] = cutpoints
        kappa=minimize_quadratic_weighted_kappa(cutpoints,Y,labels_valid) 
        print kappa
    #in this section I will get the result of training 
    print cutpoints
    ftrl.fit(grd_enc.transform(grd.apply(train[train_features])),train['Response'],validation_set=False)
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    Y_test = ftrl.predict(grd_enc.transform(grd.apply(test[train_features])))
    print cutpoints
    y_pred = pd.cut(Y_test,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    id_test = test['Id']
    out_put = pd.DataFrame({'id':id_test,'Response':y_pred})
    out_put.to_csv("../../result/ftrl.csv",index=False) 
    
    
        

  
    
        
    
   
    
    
    
    
   
    
    
    
    
    