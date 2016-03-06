import sys
import csv
import os
import cPickle
import pandas as pd
import xgboost as xgb
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
## keras
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
"""
## cutomized module
from model_library_config_1 import feat_folders, feat_names, param_spaces, int_feat
from ml_metrics import quadratic_weighted_kappa
from utils import *


global trial_counter
global log_handler


## libfm
libfm_exe = "../../libfm-1.40.windows/libfm.exe"

## rgf
call_exe = "../../rgf1.2/test/call_exe.pl"
rgf_exe = "../../rgf1.2/bin/rgf.exe"

output_path = "../../Output"
numOfClass = 8
### global params
## you can use bagging to stabilize the predictions
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size= 1

ebc_hard_threshold = False
verbose_level = 1

def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist

#### warpper for hyperopt for logging the training reslut
# adopted from
#
def hyperopt_wrapper(param,feat_name):
    global trial_counter
    global log_handler
    trial_counter += 1

    # convert integer feat
    for f in int_feat:
        if param.has_key(f):
            param[f] = int(param[f])

    print("------------------------------------------------------------")
    print "Trial %d" % trial_counter

    print("        Model")
    print("              %s" % feat_name)
    print("        Param")
    for k,v in sorted(param.items()):
        print("              %s: %s" % (k,v))
    print("        Result")

    ## evaluate performance
    kappa_cv_mean, kappa_cv_std = hyperopt_obj(param,feat_name, trial_counter)

    ## log
    var_to_log = [
        "%d" % trial_counter,
        "%.6f" % kappa_cv_mean, 
        "%.6f" % kappa_cv_std
    ]
    for k,v in sorted(param.items()):
        var_to_log.append("%s" % v)
    writer.writerow(var_to_log)
    log_handler.flush()

    return {'loss': -kappa_cv_mean, 'attachments': {'std': kappa_cv_std}, 'status': STATUS_OK}
#### train CV and final model with a specified parameter setting
def hyperopt_obj(param, feat_name, trial_counter):
    kappa_cv = []
    cols = list(test.columns)
    #features = ["Medical_Keyword_%d"%(i) for i in range(1,49)]
    #cols = list(set(cols) - set(features))
    cols.remove('Id')
    #cols.remove('Product_Info_2')
    #feas =['Medical_Keyword_16','Medical_Keyword_36','Medical_History_35','Medical_Keyword_34','Medical_Keyword_33','Medical_Keyword_32','Medical_Keyword_41','Insurance_History_3','Medical_Keyword_28','Medical_Keyword_21','Medical_Keyword_17','Medical_Keyword_7','Medical_History_10','Medical_Keyword_18','Medical_Keyword_20','Medical_Keyword_45','Medical_History_38','Medical_Keyword_29','Medical_Keyword_43','Medical_Keyword_46','Medical_Keyword_26','Medical_History_32','Medical_Keyword_13','Medical_Keyword_44','Medical_Keyword_27','Medical_Keyword_12','Product_Info_5','Medical_Keyword_5','Medical_Keyword_2','Medical_Keyword_6','Medical_Keyword_39','Medical_Keyword_19','Medical_Keyword_14','Medical_Keyword_35','Medical_Keyword_8']
    #cols = list(set(cols) - set(feas))
    """
    cols.remove('Family_Hist_5')
    cols.remove('Family_Hist_3')
    cols.remove('Medical_History_1')
    cols.remove('Employment_Info_4')
    cols.remove('Medical_History_26')
    cols.remove('Insurance_History_4')
    cols.remove('Medical_Keyword_32')
    cols.remove('Medical_History_25')
    """
    train_features = cols
    for run in range(1,4):
        print("run%d"%(run))
        rng = np.random.RandomState(2015 + 1000 * run)
        #### all the path
        #load index 
        path = "../../data/info/run%d"%(run)
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
        X_train = train.iloc[train_index][train_features]
        X_valid = train.iloc[test_index][train_features]
        
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        X_train = imp.fit_transform(X_train)
        X_valid = imp.fit_transform(X_valid)
        labels_train = train.iloc[train_index]['Response']
        labels_valid = train.iloc[test_index]['Response']
        
        # cdf
        cdf_valid_path = "%s/valid.cdf" % path
        ## load cdf
        cdf_valid = np.loadtxt(cdf_valid_path, dtype=float)
        w = np.loadtxt("../../data/info/All/weight.txt",dtype=float)

        ## make evalerror func
        evalerror_regrank_valid = lambda preds,dtrain: evalerror_regrank_cdf(preds, dtrain, cdf_valid)
        evalerror_softmax_valid = lambda preds,dtrain: evalerror_softmax_cdf(preds, dtrain, cdf_valid)
        evalerror_softkappa_valid = lambda preds,dtrain: evalerror_softkappa_cdf(preds, dtrain, cdf_valid)
        evalerror_ebc_valid = lambda preds,dtrain: evalerror_ebc_cdf(preds, dtrain, cdf_valid, ebc_hard_threshold)
        evalerror_cocr_valid = lambda preds,dtrain: evalerror_cocr_cdf(preds, dtrain, cdf_valid)

        ##############
        ## Training ##
        ##############
        ## you can use bagging to stabilize the predictions    
        if param.has_key("booster"):
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
            dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
            watchlist = []
            if verbose_level >= 2:
                watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]   
   
        ## various models
        
        if param["task"] in ["regression", "ranking"]:
            ## regression & pairwise ranking with xgboost
            watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
            bst = xgb.train(param, dtrain_base, param['num_round'],watchlist,feval=evalerror_regrank_valid)
            pred = bst.predict(dvalid_base)
        elif param["task"] in ["softmax"]:
            ## softmax regression with xgboost
            
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid-1,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w[train_index])
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=evalerror_softmax_valid)
            pred = bst.predict(dvalid_base)
            w = np.asarray(range(1,numOfClass+1))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)
        elif param['task'] in['class']:
            labels_train = train.iloc[train_index]['Response'] - 1
            labels_valid = train.iloc[test_index]['Response'] - 1
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w[train_index])
            watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
            bst = xgb.train(param, dtrain_base, param['num_round'],watchlist)
            pred = bst.predict(dvalid_base)
            
        elif param["task"] in ["softkappa"]:
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid-1,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w[train_index])
            ## softkappa with xgboost
            obj = lambda preds, dtrain: softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_softkappa_valid)
            pred = softmax(bst.predict(dvalid_base))
            w = np.asarray(range(1,numOfClass+1))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)

        elif param["task"]  in ["ebc"]:
            ## ebc with xgboost
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid-1,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w[train_index])
            obj = lambda preds, dtrain: ebcObj(preds, dtrain)
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_ebc_valid)
            pred = sigmoid(bst.predict(dvalid_base))
            pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)

        elif param["task"]  in ["cocr"]:
            ## cocr with xgboost
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid-1,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w[train_index])
            obj = lambda preds, dtrain: cocrObj(preds, dtrain)
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_cocr_valid)
            pred = bst.predict(dvalid_base)
            pred = applyCOCRRule(pred)

        elif param['task'] == "reg_skl_rf":
            ## regression with sklearn random forest regressor
            rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                       max_features=param['max_features'],
                                       n_jobs=param['n_jobs'],
                                       random_state=param['random_state'])
            rf.fit(X_train, labels_train)
            train_sort = pd.DataFrame({'cols':train_features,'value':list(rf.feature_importances_)}).sort(columns=['value'],ascending=False)
            train_sort.to_csv("sort.csv")
            pred = rf.predict(X_valid)

        elif param['task'] == "reg_skl_etr":
            ## regression with sklearn extra trees regressor
            etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
            etr.fit(X_train, labels_train)
            pred = etr.predict(X_valid)

        elif param['task'] == "reg_skl_gbm":
            ## regression with sklearn gradient boosting regressor
            gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                            max_features=param['max_features'],
                                            learning_rate=param['learning_rate'],
                                            max_depth=param['max_depth'],
                                            subsample=param['subsample'],
                                            random_state=param['random_state'])
            gbm.fit(X_train, labels_train)
            pred = gbm.predict(X_valid.toarray())

        elif param['task'] == "clf_skl_lr":
            ## classification with sklearn logistic regression
            lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                    C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=param['random_state'])
            lr.fit(X_train, labels_train)
            pred = lr.predict_proba(X_valid)
            w = np.asarray(range(1,numOfClass+1))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)
            pred = pred - 1

        elif param['task'] == "reg_skl_svr":
            ## regression with sklearn support vector regression
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                    degree=param['degree'], kernel=param['kernel'])
            svr.fit(X_train, labels_train)
            pred = svr.predict(X_valid)

        elif param['task'] == "reg_skl_ridge":
            ## regression with sklearn ridge regression
            ridge = Ridge(alpha=param["alpha"], normalize=True)
            ridge.fit(X_train, labels_train)
            pred = ridge.predict(X_valid)
        elif param['task'] == "reg_skl_lasso":
            ## regression with sklearn lasso
            lasso = Lasso(alpha=param["alpha"], normalize=True)
            lasso.fit(X_train, labels_train)
            pred = lasso.predict(X_valid)
        """
        elif param['task'] == "reg_keras_dnn":
            ## regression with keras' deep neural networks
            model = Sequential()
            ## input layer
            model.add(Dropout(param["input_dropout"]))
            ## hidden layers
            first = True
            hidden_layers = param['hidden_layers']
            while hidden_layers > 0:
                if first:
                    dim = X_train.shape[1]
                    first = False
                else:
                    dim = param["hidden_units"]
                model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
                if param["batch_norm"]:
                    model.add(BatchNormalization((param["hidden_units"],)))
                if param["hidden_activation"] == "prelu":
                    model.add(PReLU((param["hidden_units"],)))
                else:
                    model.add(Activation(param['hidden_activation']))
                model.add(Dropout(param["hidden_dropout"]))
                hidden_layers -= 1

            ## output layer
            model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
            model.add(Activation('linear'))

            ## loss
            model.compile(loss='mean_squared_error', optimizer="adam")

            ## to array
            X_train = X_train.toarray()
            X_valid = X_valid.toarray()

            ## scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

            ## train
            model.fit(X_train, labels_train+1,
                        nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
                        validation_split=0, verbose=0)

            ##prediction
            pred = model.predict(X_valid, verbose=0)
            pred.shape = (X_valid.shape[0],labels_valid)
        """
        """
        elif param['task'] == "reg_rgf":
            ## regression with regularized greedy forest (rgf)
            ## to array
            X_train, X_valid = X_train.toarray(), X_valid.toarray()

            train_x_fn = feat_train_path+".x"
            train_y_fn = feat_train_path+".y"
            valid_x_fn = feat_valid_path+".x"
            valid_pred_fn = feat_valid_path+".pred"

            model_fn_prefix = "rgf_model"

            np.savetxt(train_x_fn, X_train[index_base], fmt="%.6f", delimiter='\t')
            np.savetxt(train_y_fn, labels_train[index_base], fmt="%d", delimiter='\t')
            np.savetxt(valid_x_fn, X_valid, fmt="%.6f", delimiter='\t')
            # np.savetxt(valid_y_fn, labels_valid, fmt="%d", delimiter='\t')


            pars = [
                "train_x_fn=",train_x_fn,"\n",
                "train_y_fn=",train_y_fn,"\n",
                #"train_w_fn=",weight_train_path,"\n",
                "model_fn_prefix=",model_fn_prefix,"\n",
                "reg_L2=", param['reg_L2'], "\n",
                #"reg_depth=", 1.01, "\n",
                "algorithm=","RGF","\n",
                "loss=","LS","\n",
                #"opt_interval=", 100, "\n",
                "valid_interval=", param['max_leaf_forest'],"\n",
                "max_leaf_forest=", param['max_leaf_forest'],"\n",
                "num_iteration_opt=", param['num_iteration_opt'], "\n",
                "num_tree_search=", param['num_tree_search'], "\n",
                "min_pop=", param['min_pop'], "\n",
                "opt_interval=", param['opt_interval'], "\n",
                "opt_stepsize=", param['opt_stepsize'], "\n",
                "NormalizeTarget"
            ]
            pars = "".join([str(p) for p in pars])

            rfg_setting_train = "./rfg_setting_train"
            with open(rfg_setting_train+".inp", "wb") as f:
                f.write(pars)

            ## train fm
            cmd = "perl %s %s train %s >> rgf.log" % (
                    call_exe, rgf_exe, rfg_setting_train)
            #print cmd
            os.system(cmd)


            model_fn = model_fn_prefix + "-01" 
            pars = [
                "test_x_fn=",valid_x_fn,"\n",
                "model_fn=", model_fn,"\n",
                "prediction_fn=", valid_pred_fn
            ]

            pars = "".join([str(p) for p in pars])
            
            rfg_setting_valid = "./rfg_setting_valid"
            with open(rfg_setting_valid+".inp", "wb") as f:
                f.write(pars)
            cmd = "perl %s %s predict %s >> rgf.log" % (
                    call_exe, rgf_exe, rfg_setting_valid)
            #print cmd
            os.system(cmd)

            pred = np.loadtxt(valid_pred_fn, dtype=float)
        """
        ## weighted averageing over different models
        if 'class' in param['task']:
            kappa_valid = quadratic_weighted_kappa(pred+1, labels_valid+1)
            kappa_cv.append(kappa_valid)
        else:
            pred_raw = pred 
            pred_rank = pred_raw.argsort().argsort()
            pred_score = getScore(pred_rank, cdf_valid, valid=False)
            #pred_score = getScore_1(pred)
            kappa_valid = quadratic_weighted_kappa(pred_score, labels_valid)
            kappa_cv.append(kappa_valid)
        print("kappa_valid%f"%(kappa_valid))
        
    
    ## save this prediction
    #dfPred = pd.DataFrame({"target": labels_valid, "prediction": pred_raw})
    #dfPred.to_csv(raw_pred_valid_path, index=False, header=True,
    #             columns=["target", "prediction"])

    kappa_cv_mean = np.mean(kappa_cv)
    kappa_cv_std = np.std(kappa_cv)
    if verbose_level >= 1:
        print("              Mean: %.6f" % kappa_cv_mean)
        print("              Std: %.6f" % kappa_cv_std)
    ####################
    #### Retraining ####
    ####################
    #### all the path
    
    output_path = "../../result"
    #path = "%s/All" % (feat_folder)
    save_path = "%s/All" % output_path
    subm_path = "%s/Subm" % output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(subm_path):
        os.makedirs(subm_path)
    
    # cdf
    cdf_test_path = "../../data/info/All/test.cdf"
    # raw prediction path (rank)
    raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    # submission path (relevance as in [1,2,3,4])
    subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)

    X_train = train[train_features]
    X_valid = test[train_features]
    labels_train = train['Response']
    ## load cdf
    cdf_test = np.loadtxt(cdf_test_path, dtype=float)  
    ##
    evalerror_regrank_test = lambda preds,dtrain: evalerror_regrank_cdf(preds, dtrain, cdf_test)
    evalerror_softmax_test = lambda preds,dtrain: evalerror_softmax_cdf(preds, dtrain, cdf_test)
    evalerror_softkappa_test = lambda preds,dtrain: evalerror_softkappa_cdf(preds, dtrain, cdf_test)
    evalerror_ebc_test = lambda preds,dtrain: evalerror_ebc_cdf(preds, dtrain, cdf_test, ebc_hard_threshold)
    evalerror_cocr_test = lambda preds,dtrain: evalerror_cocr_cdf(preds, dtrain, cdf_test)
    if param.has_key("booster"):
            dvalid_base = xgb.DMatrix(X_valid)
            dtrain_base = xgb.DMatrix(X_train, label=labels_train)    
            watchlist = []
            if verbose_level >= 2:
                watchlist  = [(dtrain_base, 'train')]
    if param["task"] in ["regression", "ranking"]:
        ## regression & pairwise ranking with xgboost
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train -1) 
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=evalerror_regrank_valid)
        pred = bst.predict(dvalid_base)
        
    elif param["task"] in ["softmax"]:
        ## softmax regression with xgboost
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=evalerror_softmax_valid)
        pred = bst.predict(dvalid_base)
        w = np.asarray(range(1,numOfClass+1))
        pred = pred * w[np.newaxis,:]
        pred = np.sum(pred, axis=1)

    elif param["task"] in ["softkappa"]:
        ## softkappa with xgboost
        obj = lambda preds, dtrain: softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_softkappa_valid)
        pred = softmax(bst.predict(dvalid_base))
        w = np.asarray(range(1,numOfClass+1))
        pred = pred * w[np.newaxis,:]
        pred = np.sum(pred, axis=1)

    elif param["task"]  in ["ebc"]:
        ## ebc with xgboost
        obj = lambda preds, dtrain: ebcObj(preds, dtrain)
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_ebc_valid)
        pred = sigmoid(bst.predict(dvalid_base))
        pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)

    elif param["task"]  in ["cocr"]:
        ## cocr with xgboost
        obj = lambda preds, dtrain: cocrObj(preds, dtrain)
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_cocr_valid)
        pred = bst.predict(dvalid_base)
        pred = applyCOCRRule(pred)

    elif param['task'] == "reg_skl_rf":
        ## regression with sklearn random forest regressor
        rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                   max_features=param['max_features'],
                                   n_jobs=param['n_jobs'],
                                   random_state=param['random_state'])
        rf.fit(X_train, labels_train+1)
        pred = rf.predict(X_valid)
        print rf.feature_importances_
        print train_features

    elif param['task'] == "reg_skl_etr":
        ## regression with sklearn extra trees regressor
        etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                  max_features=param['max_features'],
                                  n_jobs=param['n_jobs'],
                                  random_state=param['random_state'])
        etr.fit(X_train, labels_train)
        pred = etr.predict(X_valid)

    elif param['task'] == "reg_skl_gbm":
        ## regression with sklearn gradient boosting regressor
        gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                        max_features=param['max_features'],
                                        learning_rate=param['learning_rate'],
                                        max_depth=param['max_depth'],
                                        subsample=param['subsample'],
                                        random_state=param['random_state'])
        gbm.fit(X_train, labels_train)
        pred = gbm.predict(X_valid.toarray())

    elif param['task'] == "clf_skl_lr":
        ## classification with sklearn logistic regression
        lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                class_weight='auto', random_state=param['random_state'])
        lr.fit(X_train, labels_train)
        pred = lr.predict_proba(X_valid)
        w = np.asarray(range(1,numOfClass+1))
        pred = pred * w[np.newaxis,:]
        pred = np.sum(pred, axis=1)

    elif param['task'] == "reg_skl_svr":
        ## regression with sklearn support vector regression
        X_train, X_valid = X_train, X_valid
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                degree=param['degree'], kernel=param['kernel'])
        svr.fit(X_train, labels_train)
        pred = svr.predict(X_valid)

    elif param['task'] == "reg_skl_ridge":
        ## regression with sklearn ridge regression
        ridge = Ridge(alpha=param["alpha"], normalize=True)
        ridge.fit(X_train, labels_train)
        pred = ridge.predict(X_valid)

    elif param['task'] == "reg_skl_lasso":
        ## regression with sklearn lasso
        lasso = Lasso(alpha=param["alpha"], normalize=True)
        lasso.fit(X_train, labels_train)
        pred = lasso.predict(X_valid)
    pred_raw = pred
    pred_rank = pred_raw.argsort().argsort()
    #
    ## write
    id_test = test['Id']
    if param['task'] in ['class']:
        output = pd.DataFrame({"Id": id_test, "Response": pred_score})  
    else:
        output = pd.DataFrame({"Id": id_test, "Response": pred_raw})    
        output.to_csv(raw_pred_test_path, index=False)
        output = pd.DataFrame({"Id": id_test, "Response": pred_rank})    
        output.to_csv(rank_pred_test_path, index=False)
        pred_score = getScore(pred, cdf_test)
        output = pd.DataFrame({"Id": id_test, "Response": pred_score})    
        output.to_csv(subm_path, index=False)
    return kappa_cv_mean, kappa_cv_std
 
####################
## Model Buliding ##
####################
if __name__ == "__main__":
    train = pd.read_csv("../../data/train_clean_3.csv")
    test = pd.read_csv("../../data/test_clean_3.csv")
    log_path = "%s/Log" % output_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    #feat_names = ['param_space_clf_xgb_linear']
    feat_names = ['param_space_kappa_xgb_linear']
    for feat_name in feat_names:
        param_space = param_spaces[feat_name]
   
        log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
        log_handler = open(log_file, 'wb' )
        writer = csv.writer( log_handler )
        headers = [ 'trial_counter', 'kappa_mean', 'kappa_std' ]
        for k,v in sorted(param_space.items()):
            headers.append(k)
        writer.writerow( headers )
        log_handler.flush()
        
        print("************************************************************")
        print("Search for the best params")
        #global trial_counter
        trial_counter = 0
        trials = Trials()
        objective = lambda p: hyperopt_wrapper(p,feat_name)
        best_params = fmin(objective, param_space, algo=tpe.suggest,
                           trials=trials, max_evals=param_space["max_evals"])
        for f in int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k,v in best_params.items():
            print "        %s: %s" % (k,v)
        trial_kappas = -np.asarray(trials.losses(), dtype=float)
        best_kappa_mean = max(trial_kappas)
        ind = np.where(trial_kappas == best_kappa_mean)[0][0]
        best_kappa_std = trials.trial_attachments(trials.trials[ind])['std']
        print("Kappa stats")
        print("        Mean: %.6f\n        Std: %.6f" % (best_kappa_mean, best_kappa_std))
    
