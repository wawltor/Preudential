import pandas as pd 
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from sklearn import  preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer

param_Medical_History_1 = {
    'task':'Medical_History_1',
    'alpha':1.00039818628,
    'cat':True                   
}



param_Medical_History_15 = {
    'task':'Medical_History_15',
    'alpha':1.00002601816,
    'cat':True                   
}

param_Family_Hist_2 = {
    'task':'Family_Hist_2',
    'alpha':1.00089351238 , 
    'cat':False                
}
param_Employment_Info_6 ={
    'task':'Employment_Info_6',
    'alpha':1.00001694783  ,
    'cat':False                        
}
#this function, I will  use bestparameter to predict the value of missing
def predictValue(param):
    
    train = pd.read_csv("../../data/train_all_1.csv")
    test = pd.read_csv("../../data/test_all_1.csv")
    #add price message to train
    feature = param['task']
    train_t = train[~pd.isnull(train[feature])]
    train_p = train[pd.isnull(train[feature])]
    test_t = test[~pd.isnull(test[feature])]
    test_p = test[pd.isnull(test[feature])]
    cols = list(test.columns)
    null_features = ['Employment_Info_1','Medical_History_32','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5','Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24']
    null_features.append('Id')
    null_features.append('Product_Info_2')
    train_feature = list(set(cols)-set(null_features))
    #for train_data
    ridge = Ridge(alpha=param['alpha'],normalize=True)
    ridge.fit(train_t[train_feature],train_t[feature])
    pred_value = ridge.predict(train_p[train_feature])
    if param['cat'] == True:
        pred_value = np.rint(param['cat'])
    train_p[feature] = pred_value
    #for test_data
    ridge = Ridge(alpha=param['alpha'],normalize=True)
    ridge.fit(test_t[train_feature],test_t[feature])
    pred_value = ridge.predict(test_p[train_feature])
    if param['cat'] == True:
        pred_value = np.rint(param['cat'])
    test_p[feature] = pred_value
    #fill data 
    train_p.to_csv("temp.csv")
    train[feature][pd.isnull(train[feature])]= train_p[feature]
    test[feature][pd.isnull(test[feature])]= test_p[feature]
    #save data
    train.to_csv("../../data/train_all_1.csv",index=False)
    test.to_csv("../../data/test_all_1.csv",index=False)
     

if __name__ == '__main__':
    
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    encoders = dict()
    encoders['Product_Info_2'] = preprocessing.LabelEncoder()
    train['Product_Info_2'] = encoders['Product_Info_2'].fit_transform(train['Product_Info_2'])
    test['Product_Info_2'] = encoders['Product_Info_2'].fit_transform(test['Product_Info_2'])
    cols = list(train.columns)
    imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
    train = imp.fit_transform(train)
    test  = imp.fit_transform(test)
    train = pd.DataFrame(train,columns=cols)
    cols.remove('Response')
    test = pd.DataFrame(test,columns = cols)
    train.to_csv("../../data/train_all_2.csv",index=False)
    test.to_csv("../../data/test_all_2.csv",index=False)
    
    
  
