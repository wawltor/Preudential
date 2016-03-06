import sys
reload(sys)
import pandas as pd
import json                                                                     
import urllib                                                                   
from urllib2 import  urlopen, URLError, HTTPError                       
def translate(inputFile, outputFile):
    fin = open(inputFile, 'r')                                              
    fout = open(outputFile, 'w')                                            
    
    for eachLine in fin:                                                    
        line = eachLine.strip()                                         
        quoteStr = urllib.quote(line)                                   
        url = 'http://openapi.baidu.com/public/2.0/bmt/translate?client_id=WtzfFYTtXyTocv7wjUrfGR9W&q=' + quoteStr + '&from=auto&to=zh'
        try:
            resultPage = urlopen(url)                               
        except HTTPError as e:
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        except Exception, e:
            print 'translate error.'
            print e
            continue

        resultJason = resultPage.read().decode('utf-8')                
        js = None
        try:
            js = json.loads(resultJason)                           
        except Exception, e:
            print 'loads Json error.'
            print e
            continue
    
        key = u"trans_result" 
        if key in js:
            dst = js["trans_result"][0]["dst"]                     
            outStr = dst
        else:
            outStr = line                                          

        fout.write(outStr.strip().encode('utf-8') + '\n')              
        
    fin.close()
    fout.close()

def trainslate_line(line):
    global i 
    i = i + 1
    print i
    line = line.strip()                                        
    quoteStr = urllib.quote(line)                                   
    url = 'http://openapi.baidu.com/public/2.0/bmt/translate?client_id=WtzfFYTtXyTocv7wjUrfGR9W&q=' + quoteStr + '&from=auto&to=zh'
    #url = "http://openapi.baidu.com/public/2.0/translate/dict/simple?client_id=5kHZHeo8MN7L6NmPTGV6POsb&q=@word&from=en&to=zh";

    try:
        resultPage = urlopen(url)                               
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
    except Exception, e:
        print 'translate error.'
        print e
        
    resultJason = resultPage.read().decode('utf-8')                
    js = None
    try:
        js = json.loads(resultJason)                          
    except Exception, e:
        print 'loads Json error.'
        print e
    key = u"trans_result" 
    if key in js:
        dst = js["trans_result"][0]["dst"]                     
        outStr = dst
    else:
        outStr = line  
    print outStr                                       
    return outStr                                      
  
if __name__ == '__main__':
    global i 
    i = 0
    train = pd.read_csv("temp.csv")
    messages = list(train.iloc[1500:1723].apply(lambda x : trainslate_line(x['content']),axis=1))
    file = open("2.txt","a")
    for mess in messages:
        file.write(mess+"\n")
    
    """
    train  = pd.read_csv("test.csv")  
    global i 
    i =  0
    
    train.iloc[0:3]['title_cn']  = list(train.iloc[0:3].apply(lambda x : trainslate_line(x['title']),axis=1))
    #train['content_cn'] = list(train.apply(lambda x : trainslate_line(x['content']),axis=1))
    train.to_csv("temp.csv",index=False)
    """                          

