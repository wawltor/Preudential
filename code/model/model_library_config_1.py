import numpy as np
from hyperopt import hp


############
## Config ##
############

debug =False

## xgboost
xgb_random_seed = 2015
xgb_nthread = 16
xgb_dmatrix_silent = True

## sklearn
skl_random_seed = 2015
skl_n_jobs = 8
if debug:
    xgb_nthread = 1
    skl_n_jobs = 1
    xgb_min_num_round = 5
    xgb_max_num_round = 10
    xgb_num_round_step = 5
    skl_min_n_estimators = 5
    skl_max_n_estimators = 10
    skl_n_estimators_step = 5
    libfm_min_iter = 5
    libfm_max_iter = 10
    iter_step = 5
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 1
    hyperopt_param["rf_max_evals"] = 1
    hyperopt_param["etr_max_evals"] = 1
    hyperopt_param["gbm_max_evals"] = 1
    hyperopt_param["lr_max_evals"] = 1
    hyperopt_param["ridge_max_evals"] = 1
    hyperopt_param["lasso_max_evals"] = 1
    hyperopt_param['svr_max_evals'] = 1
    hyperopt_param['dnn_max_evals'] = 1
    hyperopt_param['libfm_max_evals'] = 1
    hyperopt_param['rgf_max_evals'] = 1
else:
    xgb_min_num_round = 300
    xgb_max_num_round = 500
    xgb_num_round_step = 10
    skl_min_n_estimators = 10
    skl_max_n_estimators = 500
    skl_n_estimators_step = 10
    libfm_min_iter = 10
    libfm_max_iter = 500
    iter_step = 10
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 200
    hyperopt_param["rf_max_evals"] = 200
    hyperopt_param["etr_max_evals"] = 200
    hyperopt_param["gbm_max_evals"] = 200
    hyperopt_param["lr_max_evals"] = 200
    hyperopt_param["ridge_max_evals"] = 200
    hyperopt_param["lasso_max_evals"] = 200
    hyperopt_param['svr_max_evals'] = 200
    hyperopt_param['dnn_max_evals'] = 200
    hyperopt_param['libfm_max_evals'] = 200
    hyperopt_param['rgf_max_evals'] = 200



########################################
## Parameter Space for XGBoost models ##
########################################
## In the early stage of the competition, I mostly focus on
## raw tfidf features and linear booster.

## regression with linear booster
param_space_regrank_xgb_tree_count = {
    'task': 'regrank',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    'eta': 0.0025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 500,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
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
    'num_round': 500,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}


param_space_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    'eta': 0.0025,
    #'eta':0.1,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 12000,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}

param_space_reg_xgb_tree_count_new = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
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
}

param_space_poisson_regrank_xgb_tree_count = {
    'task': 'regrank',
    'booster': 'gbtree',
    'objective': 'count:poisson',
    'max_delta_step':1,
    'eta': 0.0025,
    #'eta':0.1,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 12000,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}


param_space_poisson_reg_xgb_tree_count = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'count:poisson',
    'max_delta_step':1,
    'eta': 0.0025,
    #'eta':0.1,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 12000,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}
param_space_clf_xgb_linear = {
    'task': 'softmax',
    'booster': 'gblinear',
    'objective': 'multi:softprob',
    'max_delta_step':1,
    'eta': 0.025,
    #'eta':0.1,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 120,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
    'num_class': 8
}

param_space_rank_xgb_linear = {
    'task': 'regrank',
    'booster': 'gbtree',
    'objective': 'rank:pairwise',
    'max_delta_step':1,
    'eta': 0.025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 500,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}
param_space_kappa_xgb_linear = {
    'task': 'softkappa',
    'booster': 'gbtree',
    'objective': 'reg:linear', # for linear raw predict score
    'hess_scale': 0.0025,
    'max_delta_step':1,
    'eta': 0.025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 500,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
}




## extended binary classification (ebc) with linear booster
param_space_ebc_xgb_linear = {
    'task': 'ebc',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'max_delta_step':1,
    'eta': 0.025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 1,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0,
    'num_class': 8
    
}

## cost-sensitive ordinal classification via regression (cocr) with linear booster
param_space_cocr_xgb_linear = {
    'task': 'cocr',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'num_class': 8,
    'max_delta_step':1,
    'eta': 0.025,
    'eval_metric': 'rmse',
    'min_child_weight': 4,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_round': 500,
    'nthread': xgb_nthread,
    'num_parallel_tree':1,
    'base_score':0.5,
    'silent': 1,
    'seed': 2,
    "max_evals": 1,
    'lambda':0,
    'alpha':0,
    'lambda_bias':0
}


########################################
## Parameter Space for Sklearn Models ##
########################################

## random forest regressor
param_space_reg_skl_rf = {
    'task': 'reg_skl_rf',
    'n_estimators':300,
    'max_features': 0.28,
    'n_jobs': 64,
    'random_state': skl_random_seed,
    "max_evals": 1,
}

## extra trees regressor
param_space_reg_skl_etr = {
    'task': 'reg_skl_etr',
    'n_estimators': 500,
    'max_features': 0.4,
    'n_jobs': 32,
    'random_state': skl_random_seed,
    "max_evals": 1,
}

## gradient boosting regressor
param_space_reg_skl_gbm = {
    'task': 'reg_skl_gbm',
    'n_estimators': 400,
    'learning_rate': 0.003125,
    'max_features': 0.7,
    'max_depth': 8,
    'max_leaf_nodes':4,
    'subsample': 0.7,
    'random_state': skl_random_seed,
    "max_evals": 1,
}



## ridge regression
param_space_reg_skl_ridge = {
    'task': 'reg_skl_ridge',
    #'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'alpha':1.01,
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["ridge_max_evals"],
}

param_space_reg_skl_svr = {
    'task': 'reg_skl_svr',
    'C': 2,
    'gamma': 0.01,
    'degree': 4,
    'epsilon':0.001,    
    'kernel':  ['rbf'],
    "max_evals": 1,
}
## lasso
param_space_reg_skl_lasso = {
    'task': 'reg_skl_lasso',
    'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["lasso_max_evals"],
}

## logistic regression
param_space_clf_skl_lr = {
    'task': 'clf_skl_lr',
    'C': hp.loguniform("C", np.log(0.001), np.log(10)),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["lr_max_evals"],
}


######################################################
## Parameter Space for Factorization Machine Models ##
######################################################

## regression with libfm
param_space_reg_libfm = {
    'task': 'reg_libfm',
    'dim': hp.quniform("dim", 1, 20, 1),
    "iter": hp.quniform("iter", libfm_min_iter, libfm_max_iter, iter_step),
    "max_evals": hyperopt_param["libfm_max_evals"],
}


######################################
## Parameter Space for Keras Models ##
######################################

## regression with Keras' deep neural network
param_space_reg_keras_dnn = {
    'task': 'reg_keras_dnn',
    'batch_norm': hp.choice("batch_norm", [True, False]),
    "hidden_units": hp.choice("hidden_units", [64, 128, 256, 512]),
    "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4]),
    "input_dropout": hp.quniform("input_dropout", 0, 0.9, 0.1),
    "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.9, 0.1),
    "hidden_activation": hp.choice("hidden_activation", ["relu", "prelu"]),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "nb_epoch": hp.choice("nb_epoch", [10, 20, 30, 40]),
    "max_evals": 400,
}





## integer features
int_feat = ["num_round", "n_estimators", "max_depth", "degree",
            "hidden_units", "hidden_layers", "batch_size", "nb_epoch",
            "dim", "iter",
            "max_leaf_forest", "num_iteration_opt", "num_tree_search", "min_pop", "opt_interval"]


####################
## All the Models ##
####################
feat_folders = []
feat_names = []
param_spaces = {}


#####################################
## [Feat@LSA_and_stats_feat_Jun09] ##
#####################################
#############
## xgboost ##
#############
## regression with xgboost tree booster
feat_name = "param_space_reg_xgb_tree_count"
param_spaces[feat_name] = param_space_reg_xgb_tree_count

feat_name = "param_space_regrank_xgb_tree_count"
param_spaces[feat_name] = param_space_regrank_xgb_tree_count

feat_name = "param_space_poisson_reg_xgb_tree_count"
param_spaces[feat_name] = param_space_poisson_reg_xgb_tree_count

feat_name = "param_space_poisson_regrank_xgb_tree_count"
param_spaces[feat_name] = param_space_poisson_regrank_xgb_tree_count

feat_name = "param_space_clf_xgb_linear"
param_spaces[feat_name] = param_space_clf_xgb_linear

feat_name = "param_space_rank_xgb_linear"
param_spaces[feat_name] = param_space_rank_xgb_linear

feat_name = "param_space_kappa_xgb_linear"
param_spaces[feat_name] = param_space_kappa_xgb_linear

feat_name = "param_space_ebc_xgb_linear"
param_spaces[feat_name] =  param_space_ebc_xgb_linear

feat_name = "param_space_cocr_xgb_linear"
param_spaces[feat_name] = param_space_cocr_xgb_linear

feat_name = "param_space_reg_skl_rf"
param_spaces[feat_name] = param_space_reg_skl_rf

feat_name = "param_space_reg_skl_etr"
param_spaces[feat_name] = param_space_reg_skl_etr

feat_name = "param_space_reg_skl_gbm"
param_spaces[feat_name] = param_space_reg_skl_gbm

feat_name = "param_space_reg_keras_dnn"
param_spaces[feat_name] = param_space_reg_keras_dnn

feat_name = "param_space_huber_xgb_tree_count"
param_spaces[feat_name] = param_space_huber_xgb_tree_count

feat_name = "param_space_reg_xgb_tree_count_new"
param_spaces[feat_name] = param_space_reg_xgb_tree_count_new
