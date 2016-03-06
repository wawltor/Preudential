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
random.seed(21)
np.random.seed(21)
from scipy.optimize import *
from GAOptime import *
import time
param_space_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': 0.045,
    'eval_metric': 'rmse',
    'min_child_weight': 50,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'num_round': 20,
    'nthread': 32,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
}

param_space_huber_xgb_tree_count = {
    'task': 'huber',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    'eta': 0.0025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 12000,
    'nthread': 32,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}
train = pd.read_csv("../../data/train_clean_2.csv")
test = pd.read_csv("../../data/test_clean_2.csv")
cols = list(test.columns)
cols.remove('Id')
train_features = cols
X_test_shape = test.shape[0]

LINES = 59381
start = 59381*0.2
def majority(train):
    allPreds = np.bincount(train)
    return allPreds.argmax()

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1], X[:, -1]
    y =labels
    return X[start+1:LINES], y[start+1:LINES]

def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:], X[:, 0]
    return X, ids



result = np.zeros([X_test_shape,1])
kappa_cv = []
rounds = []
param = param_space_huber_xgb_tree_count
print time.localtime()
for run in [12000]:
    param['num_round'] = run
    for jj in xrange(1):
        print(jj)
        X_train,X_valid,Y_train,Y_valid=train_test_split(train[train_features],train['Response'],test_size=0.2,random_state=jj)
        X_train = xgb.DMatrix(X_train, label=Y_train) 
        X_valid = xgb.DMatrix(X_valid,label=Y_valid)
        watchlist  = [(X_train, 'train'),(X_valid,'valid')] 
        param['seed'] = jj
        bst = xgb.train(param, X_train,param['num_round'],watchlist,early_stopping_rounds=20) 
        best_round = bst.best_iteration + 1
        xgb_test = xgb.DMatrix(test[train_features])
        pred = bst.predict(X_valid,ntree_limit=best_round)
        Y_test = bst.predict(xgb_test,ntree_limit=best_round)
        Y_train_preds = bst.predict(X_train,ntree_limit=best_round)
        cutpoints = getBestCP(Y_train,Y_train_preds)
        print 'in'
        #cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        #res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(Y_train_preds,Y_train),method='Nelder-Mead')
        #kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,Y_valid)
        #bounds = [(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10)]
        #res = differential_evolution(minimize_quadratic_weighted_kappa, bounds,args =(Y_train_preds,Y_train),strategy ='best2exp',disp=True)
        cutpoints = np.sort(cutpoints)
        score = minimize_quadratic_weighted_kappa(cutpoints,pred,Y_valid)
        cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
        print cutpoints
        y_pred = pd.cut(Y_test,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
        result[:,jj] = list(y_pred)
        kappa_cv.append(score)
        print score
    print kappa_cv
    mean_kappa = np.mean(kappa_cv)
    features = ['Response_%d'%(i) for i in range(0,1)]
    data = pd.DataFrame(result,columns=features,dtype=np.int64)
    data['Id'] = test["Id"]
    data.to_csv("../../result/bag_raw/res_%d_%f.csv"%(run,mean_kappa),index=False)
    print time.localtime()


    

