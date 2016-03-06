'''
Created on 2016/1/16

@author: FZY
'''
import pandas as pd 
import numpy as np
from sklearn.feature_extraction import DictVectorizer
def transform(data,feat):
    data[feat] = str(data[feat]) + 's'
    return data[feat]

def getDummiesInplace(columnList, train, test):
    #Takes in a list of column names and one or two pandas dataframes
    #One-hot encodes all indicated columns inplace
    columns = []
    
    if test is not None:
        df = pd.concat([train,test], axis= 0)
    else:
        df = train
        
    for columnName in df.columns:
        index = df.columns.get_loc(columnName)
        if columnName in columnList:
            dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
            columns.append(dummies)
        else:
            columns.append(df.ix[:,index])
    df = pd.concat(columns, axis = 1)
    train = df[:train.shape[0]]
    test = df[train.shape[0]:]
    print train.shape
    print test.shape
    return train, test
if __name__ == '__main__':
    #in this section , I will use 
    train = pd.read_csv("../../data/train_clean_21.csv")
    test = pd.read_csv("../../data/test_clean_21.csv")
    categorical = ["Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3", "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", "Medical_History_8", "Medical_History_9", "Medical_History_10", "Medical_History_11", "Medical_History_12", "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41"]
    train,test = getDummiesInplace(categorical, train, test)
    print train.shape
    print test.shape
    """
    feats =(train[categorical]).T.to_dict().values()
    Dvec = DictVectorizer()
    train_tmp = Dvec.fit_transform(feats).toarray()
    train = train.drop(categorical, axis=1)
    cols = Dvec.get_feature_names()
    train_tmp = pd.DataFrame(train_tmp,columns=cols)
    train_tmp.index = train.index
    train = train.join(train_tmp)
    feats =(test[categorical]).T.to_dict().values()
    test_tmp = Dvec.transform(feats).toarray()
    test = test.drop(categorical, axis=1)
    test_tmp = pd.DataFrame(test_tmp,columns=cols)
    test_tmp.index = test.index
    test = test.join(test_tmp)
    print train.shape
    print test.shape
    """
    train.to_csv("../../data/train_clean_22.csv",index=False)
    test.to_csv("../../data/test_clean_22.csv",index=False)
    
    
    