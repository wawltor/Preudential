'''
@author: FZY
'''
import pandas as pd 
import numpy as np
def expval(df,col,y,tfilter):
    tmp = pd.DataFrame(index=df.index)
    pb = df[tfilter][y].mean()                                              # train set mean
    tmp['cnt'] = df[col].map(df[tfilter][col].value_counts()).fillna(0)     # train set count
    tmp['csm'] = df[col].map(df[tfilter].groupby(col)[y].sum()).fillna(pb)  # train set sum
    tmp.ix[tfilter,'cnt'] -= 1                                              # reduce count for train set
    tmp.ix[tfilter,'csm'] -= df.ix[tfilter,y]                               # remove current value
    tmp['exp'] = ((tmp.csm+ pb*15) / (tmp.cnt+ 15)).fillna(pb)              # calculate mean including kn-extra 'average' samples 
    np.random.seed(1)
    tmp.ix[tfilter,'exp'] *= 1+.3*(np.random.rand(len(tmp[tfilter]))-.5) # add some random noise to the train set
    return tmp.exp
if __name__ == '__main__':
    
    train = pd.read_csv("../../data/train_clean_2.csv")
    test = pd.read_csv("../../data/test_clean_2.csv")
    train_nn = pd.read_csv("../../data/train_nn.csv")
    test_nn = pd.read_csv("../../data/test_nn.csv")
    
    train_add = pd.read_csv("../../data/train_clean_21.csv")
    test_add = pd.read_csv("../../data/test_clean_21.csv")
    train['FH_all_1'] = train_add['FH_all_1']
    test['FH_all_1'] = test_add['FH_all_1']
    train['FH_24_2'] = train_add['FH_24_2']
    test['FH_24_2']= test_add['FH_24_2']
    train.to_csv("../../data/train_final.csv",index=False)
    test.to_csv("../../data/test_final.csv",index=False)
    train_nn['FH_all_1'] = train_add['FH_all_1']
    test_nn['FH_all_1'] = test_add['FH_all_1']
    train_nn['FH_24_2'] = train_add['FH_24_2']
    test_nn['FH_24_2']= test_add['FH_24_2']
    test_nn['Id'] = test['Id']
    train_nn['Id'] = train['Id']
    train_nn.to_csv("../../data/train_nn.csv",index=False)
    test_nn.to_csv("../../data/test_nn.csv",index=False)
    
    