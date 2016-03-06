'''
Created on 2015/11/4

@author: FZY
'''
import random
import pandas as pd 
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from hyperopt import hp
import numpy as np
from hyperopt.pyll_utils import hyperopt_param
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,auc
from sklearn.metrics import average_precision_score
from hyperopt import Trials,tpe,fmin
from hyperopt.base import STATUS_OK
from ml_metrics import accuracy_model
from sklearn.datasets import dump_svmlight_file
import os
import xgboost as xgb
import time
#fit the  param 
debug = False
xgb_random_seed = 2015

if debug:
    hyperopt_param = {}
    hyperopt_param['lasso_max_evals'] = 2
    hyperopt_param['ridge_max_evals'] = 2
    hyperopt_param['lr_max_evals'] = 2
    hyperopt_param["xgb_max_evals"] = 2
    xgb_min_num_round = 2
    xgb_max_num_round = 10
    xgb_nthread=4
    xgb_num_round_step = 1
else:
    hyperopt_param = {}
    hyperopt_param['ridge_max_evals'] = 400
    hyperopt_param['lasso_max_evals'] = 400
    hyperopt_param['lr_max_evals'] = 400
    hyperopt_param["xgb_max_evals"] = 400
    xgb_min_num_round = 200
    xgb_max_num_round = 400
    xgb_nthread= 8
    xgb_num_round_step = 20
    
Ridge_param= {
    'task':'skl_ridge',
    'alpha': hp.loguniform("alpha", np.log(1), np.log(40)),
    'random_state':2015,
    'max_evals':hyperopt_param['ridge_max_evals']           
}

Lasso_param = {
    'task':'skl_lasso',
    'alpha': hp.loguniform("alpha", np.log(0.00001), np.log(np.exp(1))),
    'random_state':2015,
    'max_evals':hyperopt_param['lasso_max_evals']       
}

xgb_regression_param = {
    'task': 'regression',
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'eta' : hp.quniform('eta', 0, 1, 0.1),
    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    'alpha' : hp.quniform('alpha', 0, 1, 0.05),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 0,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],                    
}

xgb_regression_param_by_tree = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta' : hp.quniform('eta', 0.1, 1.0, 0.1),
    'gamma':hp.quniform('gamma',1,2,0.1),
    'max_depth':hp.quniform('max_depth',4,15,1),
    'min_child_weight':hp.quniform('min_child_weight',1,5,1),
    'colsample_bytree':hp.quniform('colsample_bytree',0.5,1,0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 0,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],                    
}

xgb_tree_param = {
    'task': 'class',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta' : hp.quniform('eta', 0.1, 1, 0.1),
    'gamma': hp.quniform('gamma',0.1,2,0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"], 
    "num_class":  9,    
    'max_depth': hp.quniform('max_depth', 6, 12, 1),             
}

skl_lr_param = {
    'task' : 'skl_lr',
    'C' : hp.quniform('C',1,20,0.1),
    'seed':xgb_random_seed,
    'max_evals':hyperopt_param['lr_max_evals']
}



def dumpMessage(bestParams,loss,loss_std,f_name,source_name,start,end):
     
    f = open("../../data/analysis/na/%s_%s_bestPara_model_log.txt"%(f_name,source_name),"wb") 
    f.write('loss:%.6f \nStd:%.6f \n'%(loss,loss_std))
    for(key,value) in bestParams.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("start_time:%s\n"%(start))
    f.write("end_time:%s\n"%(end))
    f.close()  
def trainModel(param,data,features,feature,source_name,real_value,int_boolean):
    #we juet judge our model
    #so we do not use bagging ,just one loop of CV
   
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove(feature)
    
    #create CV
    error_cv = []
    std_cv = []
    for i in range(0,3):
        X_train,X_test,Y_train,Y_test = train_test_split(data[train_feature],data[feature],test_size=0.3,random_state=i)
        n = len(list(Y_test))
        real_error = mean_squared_error([float(real_value)]*n,Y_test)
        real_var = mean_absolute_error([float(real_value)]*n,Y_test)
        print("rmse:%f  mse:%f"%(real_error,real_var))
        if param['task'] == 'skl_ridge':
            Y_train = np.array(Y_train)
            ridge = Ridge(alpha=param['alpha'],normalize=True)
            ridge.fit(X_train,Y_train)
            pred_value = ridge.predict(X_test)
            if int_boolean == True:
                pred_value = np.rint(pred_value)
            error = -np.abs(mean_squared_error(Y_test,pred_value)-real_error)
            variance = np.abs(mean_absolute_error(Y_test,pred_value)-real_var)
            #error_train = mean_absolute_error(Y_train,pred_train)
            
        elif param['task'] == 'skl_lasso':
            lasso = Lasso(alpha=param['alpha'],normalize=True)
            lasso.fit(X_train,Y_train)
            pred_value = lasso.predict(X_test)
            pred_train = lasso.predict(X_train)
            error = mean_absolute_error(Y_test,pred_value)
            variance = explained_variance_score(Y_test,pred_value)
            error_train = mean_absolute_error(Y_train,pred_train)
            #load data
        
        elif param['task'] == 'skl_lr':
            clf = LogisticRegression(C=param['C'])
            clf.fit(X_train,Y_train)
            pred_value = clf.predict(X_test)
            pred_train = clf.predict(X_train)
            error = 1 - accuracy_model(pred_value, Y_test)
            error_train =1 - accuracy_model(pred_train, Y_train)
            variance = error_train
            
        elif param['task'] == 'regression':
            train_data = xgb.DMatrix(X_train,label=np.array(Y_train))
            valid_data = xgb.DMatrix(X_test,label=np.array(Y_test))
            watchlist = [(train_data,'train'),(valid_data,'valid')]
            bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
            valid_data = xgb.DMatrix(X_test)
            pred_value = bst.predict(valid_data)
            error = -np.abs(mean_squared_error(Y_test,pred_value)-real_error)
            variance = np.abs(mean_absolute_error(Y_test,pred_value)-real_var)
        
        elif param['task'] == 'class':
            """
            if not os.path.exists("../../data/analysis/svm/train.%s.svm.txt"%(source_name)):
                dump_svmlight_file(X_train,Y_train,"../../data/analysis/svm/train.%s.svm.txt"%(source_name))
                dump_svmlight_file(X_test,Y_test,"../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
            train_data = xgb.DMatrix("../../data/analysis/svm/train.%s.svm.txt"%(source_name))
            valid_data = xgb.DMatrix("../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
            """
            train_data = xgb.DMatrix(X_train,label=Y_train)
            valid_data = xgb.DMatrix(X_test,label=Y_test)
            watchlist = [(train_data,'train'),(valid_data,'valid')]
            bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
            valid_data = xgb.DMatrix(X_test)
            pred_value = bst.predict(valid_data)
            error = 1 - accuracy_model(pred_value, Y_test)
            variance = 0
        #print "error.train:%f error.test:%f"%(error_train,error)
        error_cv.append(error)
        std_cv.append(variance)
    
    mean_error = np.mean(error_cv)
    mean_std = np.mean(variance)
    print "mase:%f"%(mean_error)
    print "mse:%f"%(mean_std)
    return {'loss':mean_error,'attachments':{'std':mean_std},'status':STATUS_OK}


def TunningParamter(param,data,features,feature,source_name,real_value,int_boolean):
    data = data[~pd.isnull(all_data[feature])]
    print data.shape
    ISOTIMEFORMAT='%Y-%m-%d %X'
    start = time.strftime(ISOTIMEFORMAT, time.localtime())
    trials = Trials()
    objective = lambda p : trainModel(p, data, features, feature,source_name,real_value,int_boolean)
    
    best_parameters = fmin(objective, param, algo =tpe.suggest,max_evals=param['max_evals'],trials= trials)
    #now we need to get best_param
    trials_loss = np.asanyarray(trials.losses(),dtype=float)
    best_loss = min(trials_loss)
    ind = np.where(trials_loss==best_loss)[0][0]
    best_loss_std = trials.trial_attachments(trials.trials[ind])['std']
    end = time.strftime(ISOTIMEFORMAT,time.localtime())
    dumpMessage(best_parameters, best_loss, best_loss_std,param['task'],source_name,start,end)
    
    
if __name__ == "__main__":
    #at first , we could rbind the train,test data 
    train = pd.read_csv("../../data/train_clean_1.csv")
    test = pd.read_csv("../../data/test_clean_1.csv")
    all_data = pd.concat([train,test],axis=0,ignore_index=True)
    cols = list(test.columns)
    null_features = ['Employment_Info_1','Medical_History_32','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5','Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24']
   
    
    #get Medical_History_15 best parameter
    train_feature = list(set(cols)-set(null_features))
    feature = 'Medical_History_15'
    train_feature.append(feature)
    real_value =  117
    #TunningParamter(Ridge_param,all_data, cols, feature,feature,real_value)
    TunningParamter(Ridge_param,all_data, train_feature, feature,feature,real_value,True)
    """
    #Employment_Info_6
    train_feature = list(set(cols)-set(null_features))
    feature = 'Employment_Info_6'
    train_feature.append(feature)
    real_value =  0.365019
    #TunningParamter(Ridge_param,all_data, cols, feature,feature,real_value)
    TunningParamter(Ridge_param,all_data, train_feature, feature,feature,real_value)
    
    #Family_Hist_2
    train_feature = list(set(cols)-set(null_features))
    feature = 'Family_Hist_2'
    train_feature.append(feature)
    real_value =  0.474558
    #TunningParamter(Ridge_param,all_data, cols, feature,feature,real_value)
    TunningParamter(Ridge_param,all_data, train_feature, feature,feature,real_value)

    #Medical_History_1
    train_feature = list(set(cols)-set(null_features))
    feature = 'Medical_History_1'
    train_feature.append(feature)
    real_value =  7.8948595
    #TunningParamter(Ridge_param,all_data, cols, feature,feature,real_value)
    TunningParamter(Ridge_param,all_data, train_feature, feature,feature,real_value)
    
    train_feature = list(set(cols)-set(null_features))
    feature = 'Family_Hist_5'
    train_feature.append(feature)
    real_value =  0.488437
    #TunningParamter(Ridge_param,all_data, cols, feature,feature,real_value)
    TunningParamter(Ridge_param,all_data, train_feature, feature,feature,real_value)
    """









