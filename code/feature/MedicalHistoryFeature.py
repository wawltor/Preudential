'''
Created on 2015/12/27

@author: FZY
'''
import pandas as pd 
if __name__ == '__main__':
    #in this model , I will get the feature of History
    train = pd.read_csv("../../data/train_clean_21.csv")
    test = pd.read_csv("../../data/test_clean_21.csv")
    # count_features = ['Medical_History_%d'%(i) for i in [1,2,4,5,15]]
    count_features_1 = ['Medical_History_%d'%(i) for i in [1,10,15,24,32]]

    train['Medical_History_count_1'] = 0
    test['Medical_History_count_1'] = 0
    for f in count_features_1:
        train['Medical_History_count_1'] = train['Medical_History_count_1'] + train[f]
        test['Medical_History_count_1'] = test['Medical_History_count_1'] + test[f]
    all_features = ['Medical_History_%d'%(i) for i in range(1,42)]
    count_features_2 = list(set(all_features)-set(count_features_1))
    train['Medical_History_count_2'] = 0
    test['Medical_History_count_2'] = 0
    for f in count_features_2:
        train['Medical_History_count_2'] = train['Medical_History_count_2'] + train[f]
        test['Medical_History_count_2'] = test['Medical_History_count_2'] + test[f]
    train.to_csv("../../data/train_clean_21.csv",index=False)
    test.to_csv("../../data/test_clean_21.csv",index=False)   