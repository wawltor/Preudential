import csv
import os
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from matplotlib import pylab as plt
import operator
from model_library_config_1 import feat_folders, feat_names, param_spaces, int_feat
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from ml_metrics import quadratic_weighted_kappa
from utils import *
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler

global trial_counter
global log_handler


ebc_hard_threshold = False
verbose_level = 1
output_path = "../../Output"


def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def plot_xgboost_importance(bst,features,file_name="feature_importance.png"):
    ceate_feature_map(features)
    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig(file_name,bbox_inches='tight',pad_inches=1)
    plt.show()

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
    cols.remove('Id')
    train_features = cols
    for run in range(1,2):
        print("run%d"%(run))
        #### all the path
        #load index 
        path = "../../data/info/run%d"%(run)
       
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
        X_train = train.iloc[train_index][train_features]
        X_valid = train.iloc[test_index][train_features]
        labels_train = train.iloc[train_index]['Response']
        labels_valid = train.iloc[test_index]['Response']
        
        # cdf
        cdf_valid_path = "%s/valid.cdf" % path
        ## load cdf
        cdf_valid = np.loadtxt(cdf_valid_path, dtype=float)
        
        ## make evalerror func
        evalerror_regrank_valid = lambda preds,dtrain: evalerror_regrank_cdf(preds, dtrain, cdf_valid)
        evalerror_softmax_valid = lambda preds,dtrain: evalerror_softmax_cdf(preds, dtrain, cdf_valid)
        evalerror_ebc_valid = lambda preds,dtrain: evalerror_ebc_cdf(preds, dtrain, cdf_valid, ebc_hard_threshold)
        evalerror_cocr_valid = lambda preds,dtrain: evalerror_cocr_cdf(preds, dtrain, cdf_valid)
        ##############
        ## Training ##
        ##############
        ## you can use bagging to stabilize the predictions    
        dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
        watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'eval')]   
        w = np.loadtxt("../../data/info/All/weight.txt",dtype=float)
        if param["task"] in ["regression", "ranking"]:
            ## regression & pairwise ranking with xgboost
            bst = xgb.train(param, dtrain_base, param['num_round'],watchlist)
            pred = bst.predict(dvalid_base)
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = bst.predict(dtrain_base)
            
        elif param["task"] in ["huber"]:
            ## regression & pairwise ranking with xgboost
            bst = xgb.train(param, dtrain_base, param['num_round'],watchlist)
            pred = bst.predict(dvalid_base)
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = bst.predict(dtrain_base)   
        elif param["task"] in ["regrank"]:
            
            bst = xgb.train(param, dtrain_base, param['num_round'],watchlist,feval=evalerror_regrank_valid)
            pred = bst.predict(dvalid_base)
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = bst.predict(dtrain_base)
        
        elif param["task"] in ["softmax"]:
            ## softmax regression with xgboost
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid-1,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w[train_index])
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=evalerror_softmax_valid)
            pred = bst.predict(dvalid_base)
            we = np.asarray(range(1,9))
            pred = np.sum(pred, axis=1)
            pred = pred + 1
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = bst.predict(dtrain_base)
            y_train_preds = y_train_preds * we[np.newaxis,:]
            y_train_preds = np.sum(y_train_preds, axis=1)
            y_train_preds = y_train_preds + 1
            
        elif param["task"]  in ["ebc"]:
            ## ebc with xgboost
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w[train_index])
            obj = lambda preds, dtrain: ebcObj(preds, dtrain)
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_ebc_valid)
            pred = sigmoid(bst.predict(dvalid_base))
            pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = sigmoid(bst.predict(dtrain_base))
            y_train_preds = applyEBCRule(y_train_preds, hard_threshold=ebc_hard_threshold)

        elif param["task"]  in ["cocr"]:
            ## cocr with xgboost
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid,weight=w[test_index])
            dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w[train_index])
            obj = lambda preds, dtrain: cocrObj(preds, dtrain)
            bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_cocr_valid)
            pred = bst.predict(dvalid_base)
            pred = applyCOCRRule(pred)
            dtrain_base = xgb.DMatrix(X_train)
            y_train_preds = bst.predict(dtrain_base)
            y_train_preds = applyCOCRRule(y_train_preds)
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
            y_train_preds = rf.predict(X_train)
        elif param['task'] == "reg_skl_etr":
        ## regression with sklearn extra trees regressor
            etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
            etr.fit(X_train, labels_train)
            pred = etr.predict(X_valid)
            y_train_preds = etr.predict(X_train)
            
        elif param['task'] == "reg_skl_gbm":
        ## regression with sklearn gradient boosting regressor
            gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                            max_features=param['max_features'],
                                            learning_rate=param['learning_rate'],
                                            max_depth=param['max_depth'],
                                            subsample=param['subsample'],
                                            random_state=param['random_state'])
            gbm.fit(X_train, labels_train)
            pred = gbm.predict(X_valid)
            y_train_preds = gbm.predict(X_train)
        elif param['task'] == "reg_skl_svr":
        ## regression with sklearn support vector regression
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                    degree=param['degree'], kernel=param['kernel'])
            svr.fit(X_train, labels_train)
            pred = svr.predict(X_valid)
        pred_raw = pred
        pred_rank = pred_raw.argsort().argsort()
        score =  getScore(pred_rank,cdf=cdf_valid)
        print quadratic_weighted_kappa(score,labels_valid)
        id_test = train.iloc[test_index]['Id']
        output = pd.DataFrame({"Id": id_test, "Response_raw": pred_raw})    
        output['Response_rank'] = pred_rank 
        output['Response_cdf'] = score
        output['Response'] = labels_valid
        ## weighted averageing over different models
        cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
        res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,labels_train),method='Nelder-Mead')
        cutpoints = np.sort(res.x)
        for i in range(0,3):
            res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,labels_train),method='Nelder-Mead')
            kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,labels_valid)
            cutpoints = np.sort(res.x)
        kappa_cv.append(kappa)
        print "kappa:%f"%(kappa)
    kappa_cv_mean = np.mean(kappa_cv)
    kappa_cv_std = np.std(kappa_cv)
    subm_path = "%s/train.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % ("../../result/All", feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
    output.to_csv(subm_path,index=False)
    print("              Mean: %.6f" % kappa_cv_mean)
    print("              Std: %.6f" % kappa_cv_std)
    
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
     
    dvalid_base = xgb.DMatrix(X_valid)
    dtrain_base = xgb.DMatrix(X_train, label=labels_train)    
    watchlist  = [(dtrain_base, 'train')]
    if param["task"] in ["regression", "ranking"]:
        ## regression & pairwise ranking with xgboost
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist)
        pred = bst.predict(dvalid_base)
        dtrain_base = xgb.DMatrix(X_train) 
        y_train_preds = bst.predict(dtrain_base)
        
    elif param["task"] in ["regrank"]:
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist,feval=evalerror_regrank_test)
        pred = bst.predict(dvalid_base)
        dtrain_base = xgb.DMatrix(X_train) 
        y_train_preds = bst.predict(dtrain_base)
        
    elif param["task"] in ["softmax"]:
            ## softmax regression with xgboost
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train-1,weight=w)
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=evalerror_softmax_test)
        pred = bst.predict(dvalid_base)
        w = np.asarray(range(1,9))
        pred = pred * w[np.newaxis,:]
        pred = np.sum(pred, axis=1)
                 
    elif param["task"] in ["softkappa"]:
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w)
        ## softkappa with xgboost
        obj = lambda preds, dtrain: softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_softkappa_test)
        pred = softmax(bst.predict(dvalid_base))
        w = np.asarray(range(1,9))
        pred = pred * w[np.newaxis,:]
        pred = np.sum(pred, axis=1)
      

    elif param["task"]  in ["ebc"]:
        ## ebc with xgboost
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w)
        obj = lambda preds, dtrain: ebcObj(preds, dtrain)
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_ebc_test)
        pred = sigmoid(bst.predict(dvalid_base))
        pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)
        
    elif param["task"]  in ["cocr"]:
        ## cocr with xgboost
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train,weight=w)
        obj = lambda preds, dtrain: cocrObj(preds, dtrain)
        bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, obj=obj, feval=evalerror_cocr_test)
        pred = bst.predict(dvalid_base)
        pred = applyCOCRRule(pred)
    elif param['task'] == "reg_skl_rf":
            ## regression with sklearn random forest regressor
        rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                   max_features=param['max_features'],
                                   n_jobs=param['n_jobs'],
                                   random_state=param['random_state'])
        rf.fit(X_train, labels_train)
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
                                        random_state=param['random_state'],
                                        max_leaf_nodes = param['max_leaf_nodes']
                                        
                                        )
        gbm.fit(X_train, labels_train)
        pred = gbm.predict(X_valid)
    
    elif param['task'] == "reg_skl_svr":
        ## regression with sklearn support vector regression
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                degree=param['degree'], kernel=param['kernel'])
        svr.fit(X_train, labels_train)
        pred = svr.predict(X_valid)
    
    pred_raw = pred
    pred_rank = pred_raw.argsort().argsort()
    id_test = test['Id']
    output = pd.DataFrame({"Id": id_test, "Response_raw": pred_raw})    
    output['Response_rank'] = pred_rank 
    pred_score = getScore(pred, cdf_test)
    output['Response_cdf'] = pred_score
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    print cutpoints
    y_pred = pd.cut(pred,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    output['Response_cut']  = y_pred      
    output.to_csv(subm_path, index=False) 
    return kappa_cv_mean, kappa_cv_std
 
####################
## Model Buliding ##
####################
if __name__ == "__main__":
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv")
    log_path = "%s/Log" % output_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    #feat_names = ['param_space_reg_xgb_tree_count' ,'param_space_regrank_xgb_tree_count','param_space_poisson_reg_xgb_tree_count','param_space_poisson_reg_xgb_tree_count','param_space_poisson_reg_xgb_tree_count','param_space_poisson_regrank_xgb_tree_count','param_space_clf_xgb_linear','param_space_rank_xgb_linear','param_space_cocr_xgb_linear','param_space_ebc_xgb_linear']
    feat_names = ['param_space_reg_xgb_tree_count']
    #feat_names = ['param_space_cocr_xgb_linear','param_space_ebc_xgb_linear','param_space_rank_xgb_linear','param_space_clf_xgb_linear']
    for feat_name in feat_names:
        param_space = param_spaces[feat_name]
        log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
        log_handler = open(log_file, 'wb' )
        writer = csv.writer( log_handler )
        headers = ['trial_counter', 'kappa_mean', 'kappa_std' ]
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
    
