import pandas as pd
import numpy as np
def majority(train,colslen):
    allPreds = np.bincount(train)
    prmax = np.max(allPreds) >= colslen
    if prmax : 
        return allPreds.argmax()
    else:
        return train['Response_1']
        
   
    

if __name__ == "__main__":
    #now we read data from the the csv file
    features = ['Response_%d'%(i) for i in range(1,4)]
    data = pd.read_csv("all.csv")
    data = data.astype(np.int64)
    colslen = 2
    data['Response'] = list(data.apply(lambda x : majority(x[features],colslen),axis=1))
    feats = ['Id','Response']
    data[feats].to_csv("res_all.csv",index=False)
