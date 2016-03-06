'''


@author: FZY
'''
message_error ={}
import pandas as pd 
def count_error_message(train):
    if train['raw'] != train['pred']:
        key = str(train['raw'])+":"+str(train['pred'])
        if key in message_error:
            message_error[key] = message_error[key] + 1
        else :
            message_error[key] = 1
if __name__ == '__main__':
    message= pd.read_csv("../../result/message_0.csv")
    count = message.shape[0]
    message.apply(lambda x : count_error_message(x),axis=1)
    keys = message_error.keys()
    print keys
    values= message_error.values()
    #values = message[keys]
    print values
    res = pd.DataFrame({'key':keys,'value':values})
    res.to_csv("res.csv",index=False)
    
    