'''
Created on 2015/12/21

@author: FZY
'''
import numpy as np
import pandas as pd
from sklearn import  preprocessing
from sklearn.linear_model import Ridge

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
    
    train = pd.read_csv("../../data/train_clean_1.csv")
    test = pd.read_csv("../../data/test_clean_1.csv")
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
    train.to_csv("../../data/train_clean_1.csv",index=False)
    test.to_csv("../../data/test_clean_1.csv",index=False)
     
    
    


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

if __name__ == '__main__':
    
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    
    
    #for preduct_info_2,has value of A1,A2,so I will convert the value to label
    encoders = dict()
    encoders['Product_Info_2'] = preprocessing.LabelEncoder()
    train['Product_Info_2'] = encoders['Product_Info_2'].fit_transform(train['Product_Info_2'])
    test['Product_Info_2'] = encoders['Product_Info_2'].fit_transform(test['Product_Info_2'])
    #deal with the null value 
    
    """
    Employment_Info_4:0.1141610953
    Employment_Info_6:0.182785739546
    Insurance_History_5:0.427678887186
    Family_Hist_2:0.482578602583
    Family_Hist_3:0.576632256109
    Family_Hist_4:0.323066300669
    Family_Hist_5:0.704114110574
    Medical_History_1:0.149694346677
    Medical_History_10:0.990619895253
    Medical_History_15:0.751014634311
    Medical_History_24:0.935989626311
    Medical_History_32:0.98135767333
    """
    
    #for Medical_History_10,Medical_History_24,Medical_History_32
    #we will a label to fill it
    
    train['Medical_History_10'][pd.isnull(train['Medical_History_10'])] = -1 
    test['Medical_History_10'][pd.isnull(test['Medical_History_10'])] = -1
    train['Medical_History_24'][pd.isnull(train['Medical_History_24'])] = -1 
    test['Medical_History_24'][pd.isnull(test['Medical_History_24'])] = -1
    train['Medical_History_32'][pd.isnull(train['Medical_History_32'])] = -1 
    test['Medical_History_32'][pd.isnull(test['Medical_History_32'])] = -1
    #Employment_Info_1:0.000151783455603
    #at first I want to use ,we only use mean value to fill it 
    train['Employment_Info_1'][np.isnan(train['Employment_Info_1'])] = np.mean(train['Employment_Info_1'])
    test['Employment_Info_1'][np.isnan(test['Employment_Info_1'])] = np.mean(test['Employment_Info_1'])
    
    #Employment_Info_4:0.1141610953,I use regression to predict
    #we know that the restult 
    train['Employment_Info_4'][pd.isnull(train['Employment_Info_4'])] = np.mean(train['Employment_Info_4'])
    test['Employment_Info_4'][pd.isnull(test['Employment_Info_4'])] = np.mean(test['Employment_Info_4'])

    #Insurance_History_5
    train['Insurance_History_5'][pd.isnull(train['Insurance_History_5'])] = np.mean(train['Insurance_History_5'])
    test['Insurance_History_5'][pd.isnull(test['Insurance_History_5'])] = np.mean(test['Insurance_History_5'])
    
    #Family_Hist_3,we use the mean value to fill it 
    train['Family_Hist_3'][pd.isnull(train['Family_Hist_3'])] = np.mean(train['Family_Hist_3'])
    test['Family_Hist_3'][pd.isnull(test['Family_Hist_3'])] = np.mean(test['Family_Hist_3'])
    
    #Family_Hist_4,we use the mean value to fill it 
    train['Family_Hist_4'][pd.isnull(train['Family_Hist_4'])] = np.mean(train['Family_Hist_4'])
    test['Family_Hist_4'][pd.isnull(test['Family_Hist_4'])] = np.mean(test['Family_Hist_4'])
    
    
    #Family_Hist_5
    train['Family_Hist_5'][pd.isnull(train['Family_Hist_5'])] = np.mean(train['Family_Hist_5'])
    test['Family_Hist_5'][pd.isnull(test['Family_Hist_5'])] = np.mean(test['Family_Hist_5'])
    train.to_csv("../../data/train_clean_1.csv",index=False)
    test.to_csv("../../data/test_clean_1.csv",index=False)
    ##Medical_History_15
    predictValue(param_Family_Hist_2)
    predictValue(param_Employment_Info_6)
    predictValue(param_Medical_History_15)
    predictValue(param_Medical_History_1)
    
    
