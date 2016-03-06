'''
# in this model , I will get best feature to fill in ,MINE

@author: FZY
'''

import pandas as pd 
from minepy import MINE
def dumpMessage(messages):
    f = open("message.txt","r")
    for line in f:
        f.write(line)
    f.close
def mine_features(data,features):
    print '...'
    for X_hat_idx in features:
        features.remove(X_hat_idx)
        subset =  features
        for xi_idx in subset:
            m = MINE()
            X_hat = data[X_hat_idx].values
            xi = data[xi_idx].values
            m.compute_score(X_hat,xi)
            I_X_hat_xi = m.mic()
            if I_X_hat_xi>0.10:
                print 'I({X_hat_idx},{xi_idx}): {I_X_hat_xi}'.format(X_hat_idx=X_hat_idx,xi_idx=xi_idx,I_X_hat_xi=I_X_hat_xi)         
          
if __name__ == '__main__':
    train = pd.read_csv("../../code/model/cdf.csv")
    cols = list(train.columns)
    cols.remove('Id')
    mine_features(train, cols)