import pandas as pd 
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from scipy.optimize import minimize 
from utils import *
param_space_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    #'eta': 0.0025,
    'eta':0.1,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 600,
    'early_stopping_rounds':500,
    'nthread': 32,
    'silent': 1,
    'seed': 3,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}
if __name__ == '__main__':
    
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    categorical = ['Product_Info_1','Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41','Medical_History_1', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
    feats =train[categorical].T.to_dict().values()
    Dvec = DictVectorizer()
    train_tmp = Dvec.fit_transform(feats).toarray()
    train = train.drop(categorical, axis=1)
    cols = Dvec.get_feature_names()
    train_tmp = pd.DataFrame(train_tmp,columns=cols)
    train_tmp.index = train.index
    train = train.join(train_tmp)
    feats =test[categorical].T.to_dicts().values()
    test_tmp = Dvec.fit_transform(feats).toarray()
    test = test.drop(categorical, axis=1)
    test_tmp = pd.DataFrame(test_tmp,columns=cols)
    test_tmp.index = test.index
    test = test.join(test_tmp)
    train.fillna(-9999)
    test.fillna(-9999)
    train_features = list(set(test.columns)).remove('Id')
    kappa_cv = []
    for run in range(1,4):
        X_train,X_valid,labels_train,labels_valid=train_test_split(train[train_features],train['Response'],test_size=0.2,random_state=run)
        dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
        watchlist = []
        watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
        bst = xgb.train(param_space_reg_xgb_tree_count, dtrain_base, param_space_reg_xgb_tree_count['num_round'],watchlist)
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
    
    #train.to_csv("../../data/train_clean_2.csv")
    #test.to_csv("../../data/test_clean_2.csv")
    
    
    
    
    
    
    
  
