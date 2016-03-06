'''

@author: FZY
'''
import pandas as pd 
import numpy as np
features = ["Medical_Keyword_%d"%(i) for i in range(1,49)]

def count_keys(x):
    num = np.sum(x[features])
    return num
if __name__ == '__main__':
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv")
    print train.shape
    print test.shape
    #test = pd.read_csv("../../data/test_clean_2.csv")
    train['key_num'] = list(train.apply(lambda x : count_keys(x),axis=1))
    test['key_num'] = list(test.apply(lambda x : count_keys(x),axis=1))
    train.to_csv("../../data/train_clean_2.csv")
    test.to_csv("../../data/test_clean_2.csv")
    
    
    
    
    