'''
Created on 2015/12/27

@author: FZY
'''
import pandas as pd 
import numpy as np
if __name__ == '__main__':
    #count the feature of Insurance_History
    train = pd.read_csv("../../data/train_clean_21.csv")
    test = pd.read_csv("../../data/test_clean_21.csv")
    features = ["Insurance_History_%d"%(i) for i in [1,2,3,4,7,8,9]]
    train['Insurance_History_count'] = 0
    test['Insurance_History_count'] = 0
    for f in features:
        train['Insurance_History_count'] = train['Insurance_History_count'] + train[f]
        test['Insurance_History_count'] = test['Insurance_History_count'] + test[f]
    train.to_csv("../../data/train_clean_21.csv",index=False)
    test.to_csv("../../data/test_clean_21.csv",index=False)
    