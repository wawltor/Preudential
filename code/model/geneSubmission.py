'''

@author: FZY
'''

import pandas as pd
import xgboost as xgb
from utils import *
from sklearn.cross_validation import train_test_split
from scipy.optimize import minimize
def getNullMessage(data,name):
    rows = data.shape[0]
    messages = ""
    message = ""
    rateMessages = ""
    for col in data.columns:
        rate = data[pd.isnull(data[col])].shape[0]/float(rows)
        message = str(col) +":" +str(rate)
        if rate > 0.1 :
            rateMessages = rateMessages + ":" +message + "\n"
        messages = messages + message + '\n'
    f = open("../../data/analysis/%s.null.analysi.txt"%(name),"wb")
    f.write(messages)
    f.write("----------------------speical message--------------------\n")
    f.write(rateMessages)
    f.close()

param_space_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    'eta': 0.25,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 120,
    'nthread': 64,
    'silent': 1,
    'early.stop.round':10,
    'seed': 2015,
    "max_evals": 1,

}
        
def predictValue(train,test,param):
    cols = list(test.columns)
    cols.remove('Id')
    train_features = cols
    kappa_cv = []
    for run in range(1,4):
        X_train,X_valid,labels_train,labels_valid=train_test_split(train[train_features],train['Response'],test_size=0.2,random_state=run)
        dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
        watchlist = []
        watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
        bst = xgb.train(param, dtrain_base, param['num_round'],watchlist)
        pred = bst.predict(dvalid_base)
        dtrain_base = xgb.DMatrix(X_train)
        y_train_preds = bst.predict(dtrain_base)
        cutpoints = [1.5,2.5,3.5,4.5,5.6,6.5,7.5]
        #cutpoints = [1,2,3,4,5,6,7]
        print 'this way'
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,labels_valid)
        kappa_cv.append(kappa)
        print "kappa:%f"%(kappa)
      
if __name__ == '__main__':
    train = pd.read_csv("../../data/train_all_2.csv")
    test = pd.read_csv("../../data/test_all_2.csv")
    #predictValue(train,test,param_space_reg_xgb_tree_count)
    getNullMessage(train,'train')
