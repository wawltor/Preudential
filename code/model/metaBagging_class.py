import numpy as np
import pandas as pd
import random
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from ml_metrics import *
from utils import *
from sklearn.cross_validation import train_test_split
param_space_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta': 0.045,
    'eval_metric': 'rmse',
    'min_child_weight': 50,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'num_round': 500,
    'nthread': 32,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'num':50,
    'num_class':8,
}

train = pd.read_csv("../../data/train_clean_2.csv")
test = pd.read_csv("../../data/test_clean_2.csv")
cols = list(test.columns)
cols.remove('Id')
train_features = cols
X_test_shape = test.shape[0]
def majority(train):
    allPreds = np.bincount(train)
    return allPreds.argmax()

def load_train_data(r_seed):
    X_train,X_valid,Y_train,Y_valid=train_test_split(train[train_features],train['Response'],test_size=0.8,random_state=r_seed)
    rbm1 = ExtraTreesClassifier(n_estimators=500,
                                  max_features=0.4,
                                  n_jobs=32,
                                  random_state=jj,verbose=1).fit(X_train,Y_train)
    rbm2 = RandomForestClassifier(n_estimators=300, max_features=0.28,n_jobs=32, verbose=1,random_state=jj).fit(X_train,Y_train)
    rbm3 = GradientBoostingClassifier(n_estimators=48,max_depth=11,subsample=0.8,min_samples_leaf=5,verbose=1,random_state=jj).fit(X_train,Y_train)
    res_mean =  rbm1.predict_proba(X_valid)+rbm2.predict_proba(X_valid)+rbm3.predict_proba(X_valid)
    res_mean = res_mean /3.0
    feats = ['new_feat_%d'%(i) for i in range(1,9)]
    new_data = pd.DataFrame(res_mean,columns=feats)
    new_data.index = X_valid.index
    all_data = pd.concat([X_valid,new_data],axis=1)
    print all_data.shape
    return all_data,Y_valid,rbm1,rbm2,rbm3
    

def load_test_data(rbm1, rbm2, rbm3):
    X = test[train_features]
    res_mean = rbm1.predict_proba(X)+rbm2.predict_proba(X)+rbm3.predict_proba(X)
    res_mean = res_mean / 3.0
    feats = ['new_feat_%d'%(i) for i in range(1,9)]
    new_data = pd.DataFrame(res_mean,columns=feats)
    all_data = pd.concat([X,new_data],axis=1)
    print all_data.shape
    return all_data

param = param_space_reg_xgb_tree_count
for run in [600]:
    result = np.zeros([X_test_shape,param['num']])
    param['num_round'] = run
    for jj in range(0,param['num']):
        X, y, rbm1, rbm2, rbm3 = load_train_data(jj)
        X_test = load_test_data(rbm1, rbm2, rbm3)
        X = xgb.DMatrix(X, label=y-1) 
        watchlist  = [(X, 'train')] 
        param['seed'] = jj
        bst = xgb.train(param,X,param['num_round'],watchlist) 
        Y_train = bst.predict(X)
        X_test = xgb.DMatrix(X_test)
        Y_test = bst.predict(X_test)
        result[:,jj] = list(Y_test+1)
    features = ['Response_%d'%(i) for i in range(0,param['num'])]
    data = pd.DataFrame(result,columns=features,dtype=np.int64)
    data['Response'] = list(data.apply(lambda x : majority(x[features]),axis=1))
    feats = ['Id','Response']
    data['Id'] = test['Id']
    data.to_csv("../../result/meta_bag/res_%d.csv"%(run),index=False)



    

