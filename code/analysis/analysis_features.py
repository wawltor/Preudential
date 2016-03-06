#this model is for check model features
#in this model , I will fill the best feature


from sklearn.feature_selection import SelectPercentile
from sklearn.cross_validation import train_test_split
import pandas as pd ,numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from CVModel import loadCVIndex
params = {
    'booster':'gbtree',
    'max_depth':8,
    'min_child_weight':4,
    'eta':0.03,#0.03
    'silent':1,
    'objective':'reg:linear',
    'eval_metric':'rmse',
    "seed":2,
    'subsample':0.7,
    "colsample_bytree":0.7,
    "num_parallel_tree":1,
    "base_score":0.5,
    "alpha":0,
    "max_delta_step":1,
    "lambda":0,
    'lambda_bias':0,
    'num_boost_round':500,
    }

    
    

def rmse(y,y_pred):
    num = y.shape[0]
    rmse_error = np.sqrt(np.sum(np.abs(y,pred)**2)/(num))
    return rmse_error
    
if __name__ =="__main__":
    train = pd.read_csv("../../data/train_all_2.csv")
    features = list(train.columns)
    features.remove('Id')
    features.remove('Response')
    X = train[features]
    y = train['Response']
    scores = []
   
    for i in range(3,20):
        print 'percentile: ',i*5
        selector = SelectPercentile(f_regression,percentile = i*5)
        selector.fit(X,y)
        X_select =  selector.transform(X)
        
        pd.DataFrame(X_select)
        
        print X_select.shape[1]," features selected"
        train_index = loadCVIndex("../../data/cv/train.run1.txt")
        test_index = loadCVIndex("../../data/cv/valid.run1.txt")
        X_train = X_select[train_index]
        X_test = X_select[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        train_base = xgb.DMatrix(X_train,label=y_train)
        valid_base = xgb.DMatrix(X_test,label=y_test)
        watchlist  = [(train_base, 'train'), (valid_base, 'eval')] 
        bst = xgb.train(params, train_base, params['num_boost_round'],watchlist)
        pred = bst.predict(valid_base)
        s = np.sqrt(mean_squared_error(y_test,pred))
        scores.append(np.array([i*5,s]))
        print 'score: ',s
        
    scores = np.asarray(scores)
    np.savetxt('score.txt',scores)
    plt.figure()
    # fig, ax = plt.subplots()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(scores[:,0], scores[:,1])
    plt.show()
    