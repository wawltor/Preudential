'''

@author: FZY
'''
import pandas as pd 
from sklearn import preprocessing
def cut_words(x):
    tmp = list(x['Product_Info_2'])
    return tmp[0]

def cut_words_class(x):
    tmp = list(x['Product_Info_2'])
    return tmp[1]
    
if __name__ == '__main__':
    #in this model , I will cut product_info 
    #cut A2 to A and 2
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    #in section ,cut infomation 
    train['product_info2_type'] = list(train.apply(lambda x : cut_words(x),axis=1))
    test['product_info2_type']= list(test.apply(lambda x : cut_words(x),axis=1))
    #in section , get information
    train['product_info2_class'] = list(train.apply(lambda x : cut_words_class(x),axis=1))
    test['product_info2_class']= list(test.apply(lambda x : cut_words_class(x),axis=1))
    encoders = dict()
    encoders['product_info2_type'] = preprocessing.LabelEncoder()
    train['product_info2_type'] = encoders['product_info2_type'].fit_transform(train['product_info2_type'])
    test['product_info2_type'] = encoders['product_info2_type'].fit_transform(test['product_info2_type'])
    #save data
    train_all = pd.read_csv("../../data/train_clean_21.csv")
    test_all = pd.read_csv("../../data/test_clean_21.csv")
    train_all['product_info2_class'] = train['product_info2_class']
    test_all['product_info2_class'] = test['product_info2_class']
    train_all['product_info2_type'] = train['product_info2_type']
    test_all['product_info2_type'] = test['product_info2_type']
    train_all.to_csv("../../data/train_clean_21.csv",index=False)
    test_all.to_csv("../../data/test_clean_21.csv",index=False)
    