import os
import sys
import cPickle
import numpy as np
import pandas as pd
def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist
def gen_info(trainPath,testPath):
    ###############
    ## Load Data ##
    ###############
    ## load data
    train = pd.read_csv("../../data/%s"%(trainPath))
    test = pd.read_csv("../../data/%s"%(testPath))
    Y = train['Response'] - 1 
    #######################
    ## Generate Features ##
    #######################
    print("Generate info...")
    print("For cross-validation...")
    for run in range(0,7):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        path = "../../data/info/run%d"%(run+1)
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run+1))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run+1))
        np.savetxt("%s/train.feat.group" % path, [len(train_index)], fmt="%d")
        np.savetxt("%s/valid.feat.group" % path, [len(test_index)], fmt="%d")
        
        hist = np.bincount(Y[train_index])
        overall_cdf_valid = np.cumsum(hist) / float(sum(hist))
        np.savetxt("%s/valid.cdf" % path, overall_cdf_valid)
    print("Done.")
    print("For training and testing...")
    path = "../../data/info/All" % ()
    if not os.path.exists(path):
        os.makedirs(path)
    
    ## group
    np.savetxt("%s/train.feat.group" % (path), [train.shape[0]], fmt="%d")
    np.savetxt("%s/test.feat.group" % (path), [test.shape[0]], fmt="%d")
    ## cdf
    hist_full = np.bincount(Y)
    print (hist_full) / float(sum(hist_full))
    overall_cdf_full = np.cumsum(hist_full) / float(sum(hist_full))
    np.savetxt("%s/test.cdf" % (path), overall_cdf_full)
    print("All Done.")
    
if __name__ == "__main__":
    gen_info("train_clean_1.csv", "test_clean_1.csv")