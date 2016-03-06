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







def clip_clf(clf,X):
    # return np.clip(clf.predict(X),1e-8,10)
    return clf.predict_proba(X)

def load_data(train,test,metabagging=False,meta_rows=5000,test_size=0.2,random_state=1):
    train_features = list(test.columns)
    train_features.remove('Id')
    y = train['Response']
    train = train[train_features]
    test = test[train_features]
    X = train.values.astype(np.float32)
    X_t = test.values.astype(np.float32)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    
    if not metabagging:
        X_train = X_train[:meta_rows]
        y_train = y_train[:meta_rows]

    if metabagging:
        rf = RandomForestClassifier(n_estimators=300,bootstrap=False,random_state=random_state,n_jobs=32).fit(X_train[:meta_rows],y_train[:meta_rows])
        et = ExtraTreesClassifier(n_estimators=300,bootstrap=False,random_state=random_state,n_jobs=32).fit(X_train[:meta_rows],y_train[:meta_rows])
        gbdt = GradientBoostingClassifier(n_estimators=1,max_depth=6,subsample=0.7,max_features=0.7,min_samples_leaf=1,random_state=random_state).fit(X_train[:meta_rows],y_train[:meta_rows])
        print 'Finished training meta models'
        new_features = clip_clf(rf,X_train[meta_rows:])
        new_features += clip_clf(et,X_train[meta_rows:])
        new_features += clip_clf(gbdt,X_train[meta_rows:])
        new_features /= 3.0
        X_train = np.hstack([X_train[meta_rows:], new_features])
        y_train = y_train[meta_rows:]
        new_features = clip_clf(rf,X_test)
        new_features += clip_clf(et,X_test)
        new_features += clip_clf(gbdt,X_test)
        new_features /= 3.0
        X_test = np.hstack([X_test, new_features])
        new_features = clip_clf(rf,X_t)
        new_features += clip_clf(et,X_t)
        new_features += clip_clf(gbdt,X_t)
        new_features /= 3.0
        X_t = np.hstack([X_t, new_features])

    return X_train,X_test,y_train,y_test,X_t


if __name__ == '__main__':
    test = pd.read_csv("../../data/test_clean_2.csv")
    ids = test['Id']
    start = 20
    end = 40 
    result = np.zeros([19765,end-start])
    id_num = 0
    for i in range(start,end):
        print 'Train clf {i}'.format(i=i)
        train = pd.read_csv("../../data/train_clean_2.csv")
        test = pd.read_csv("../../data/test_clean_2.csv")
        seed=i
        np.random.seed(seed)
        X_train,X_test,y_train,y_test,X_t = load_data(train,test,metabagging=True,meta_rows=20000,test_size=0.2,random_state=seed)
        # print 'X train shape: {X_train_shape}'.format(X_train_shape=X_train.shape)
        X = np.concatenate([X_train,X_test])
        y = np.concatenate([y_train,y_test])
        
        clf = xgb.XGBRegressor(
            learning_rate=0.045,
            n_estimators=600,
            min_child_weight=50,
            max_depth=7,
            subsample=0.8,
            colsample_bytree= 0.7,
            seed=seed
        )
        
        clf.fit(
            X,
            y,
            eval_set=[(X_train, y_train),(X_test, y_test)],
        )
        y_test_preds = clf.predict(X_test).ravel()
        y_train_preds = clf.predict(X).ravel()
        cutpoints = np.array([2.93902,3.7379,4.315,4.7522,5.5363,6.1473,6.8077])
        for i in range(0,3):
            res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(y_train_preds,y),method='Nelder-Mead')
            cutpoints = np.array(np.sort(res.x))
        cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
        preds = clf.predict(X_t).ravel()
        preds = pd.cut(preds,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
        result[:,id_num] = preds
        id_num = id_num + 1
    features = ['Response_%d'%(i) for i in range(start,end)]
    data = pd.DataFrame(result,columns=features,dtype=np.int64)
    data['Id'] = test["Id"]
    data.to_csv("../../result/meta_bag/res_%d_%d.csv"%(start,end),index=False)
        