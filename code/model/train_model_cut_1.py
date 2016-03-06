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
from ml_metrics import quadratic_weighted_kappa
from utils import *
global trial_counter
global log_handler
import cPickle as cp
from GAOptime import *
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
    cols = list(test.columns)
    cols.remove('Id')
    train_features = cols
    new_features = ['key_num','scale','Ht_BMI','BMI_H32','BMI_Age','Wt_Age','FH_24_1','FH_24_2','FH_24_3','FH_24_4','FH_all_1','Medical_History_count_1','Medical_History_count_2','Insurance_History_count','product_info2_type','product_info2_class']
    train_features = list(set(train_features)-set(new_features))
    #new_features = ['scale','Fat']
    new_features_add = ['key_num']
    result = {}
    rounds = []
    for feat in new_features_add:
        copy_features = train_features
        copy_features.append(feat)
        kappa_cv = []
        cdf_cv = []
        cps = np.zeros([6,1])
        for run in range(1,7):
            print("run%d"%(run))
            #### all the path
            #load index 
            path = "../../data/info/run%d"%(run)
            train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run))
            test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run))
            X_train = train.iloc[train_index][copy_features]
            X_valid = train.iloc[test_index][copy_features]
            labels_train = train.iloc[train_index]['Response']
            labels_valid = train.iloc[test_index]['Response']
            # cdf
            cdf_valid_path = "%s/valid.cdf" % path
            ## load cdf
            cdf_valid = np.loadtxt(cdf_valid_path, dtype=float)
            ##############
            ## Training ##
            ##############
            ## you can use bagging to stabilize the predictions    
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
            dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
            watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'eval')]   
            if param["task"] in ["regression", "ranking"]:
                ## regression & pairwise ranking with xgboost
                bst = xgb.train(param, dtrain_base, num_boost_round=param['num_round'],evals=watchlist,early_stopping_rounds=150)
                best_round = bst.best_iteration + 1
                rounds.append(best_round)
                pred = bst.predict(dvalid_base)
                dtrain_base = xgb.DMatrix(X_train)
                y_train_preds = bst.predict(dtrain_base)
            ## weighted averageing over different models
            pred_raw = pred
            pred_rank = pred_raw.argsort().argsort()
            score =  getScore(pred_rank,cdf=cdf_valid)
            cdf_score =  quadratic_weighted_kappa(score,labels_valid)
            cdf_cv.append(cdf_score)
            ## weighted averageing over different models
            cutpoints = [2.8,3.8,4.5,4.9,5.5,6.2,6.8]
            #res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,labels_train),method='Nelder-Mead')
            #cutpoints = np.sort(res.x)
            cutpoints = getBestCP(y_train_preds,labels_train)
            cutpoints = np.sort(cutpoints)
            cps[run-1] = cutpoints
            kappa=minimize_quadratic_weighted_kappa(cutpoints,pred,labels_valid)
            kappa_cv.append(kappa)
            print "kappa:%f cdf:%f"%(kappa,cdf_score)
        # cdf
        cutpoints = list(np.mean(cps,axis=0))
        subm_path = "../../result/Subm"
        cdf_test_path = "../../data/info/All/test.cdf"
        # raw prediction path (rank)
        # submission path (relevance as in [1,2,3,4])
        kappa_cv_mean = np.mean(kappa_cv)
        kappa_cv_std = np.mean(cdf_cv)
        result[feat] = kappa_cv_mean
        subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" %(subm_path, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
        X_train = train[copy_features]
        X_valid = test[copy_features]
        labels_train = train['Response']
        ## load cdf
        cdf_test = np.loadtxt(cdf_test_path, dtype=float)  
        dvalid_base = xgb.DMatrix(X_valid)
        dtrain_base = xgb.DMatrix(X_train, label=labels_train)    
        watchlist  = [(dtrain_base, 'train')]
        if param["task"] in ["regression", "ranking"]:
            ## regression & pairwise ranking with xgboost
            dvalid_base = xgb.DMatrix(X_valid)
            dtrain_base = xgb.DMatrix(X_train, label=labels_train) 
            bst = xgb.train(param, dtrain_base,best_round,watchlist)
            pred = bst.predict(dvalid_base)
            dtrain_base = xgb.DMatrix(X_train) 
            y_train_preds = bst.predict(dtrain_base)
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
    with open("features.txt","wb") as f:
        cp.dump(result,f,True)
    print rounds
    print kappa_cv
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
    
    feat_names = ['param_space_reg_xgb_tree_count']
    #feat_names = ['param_space_reg_skl_rf']
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
    
