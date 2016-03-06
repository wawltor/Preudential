import sys
import numpy as np
from ml_metrics import quadratic_weighted_kappa
#sys.path.append("../")
from param_config import config
import pandas as pd 
from scipy.optimize import minimize
from sklearn.base import  BaseEstimator, TransformerMixin
import xgboost as xgb
from sklearn.metrics import mean_squared_error
######################
## Helper Functions ##
######################
#### sigmoid
class XGBoostRegressor():
    def __init__(self, num_boost_round=500, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'reg:linear'})
    
    def fit(self, X, y, num_boost_round=None):

        params = {
            'booster':'gbtree',
            'max_depth':8,
            'min_child_weight':4,
            'eta':0.25,#0.03
            'silent':1,
            'objective':'reg:linear',
            'eval_metric':'rmse',
            "seed":2,
            'subsample':0.7,
            "colsample_bytree":0.7,
            "num_parallel_tree":1,
            "base_score":0.5,
            "alpha":0,
            "max_delta_step":1,
            "lambda":0,
            'lambda_bias':0,
            'num_boost_round':116,
            }
        num_boost_round = num_boost_round or self.num_boost_round
        num_boost_round = 116
        dtrain = xgb.DMatrix(X, label=y)
        # params = self.params
        if X.shape[1]==1:
            params.update({'colsample_bytree': 1.0})
        self.clf = xgb.train(
            params,
            dtrain, 
            num_boost_round=num_boost_round,
            )
        self.fscore = self.clf.get_fscore()
        
        bb=np.zeros(dtrain.num_col())

        for ftemp, vtemp in self.fscore.items():
            bb[int(ftemp[1:])]=vtemp
    
        bb=bb/float(bb.max())
        self.coef_= bb
        
    def predict(self, X):
        dX = xgb.DMatrix(X)
        y = self.clf.predict(dX)
        return y
    def score(self, X, y):
        Y = self.predict(X)
        return self.rmse_loss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    def rmse_loss(self,y,y_pred):
        return np.sqrt(mean_squared_error(y,y_pred))


def sigmoid(score):
    p = 1. / (1. + np.exp(-score))
    return p

#### softmax
def softmax(score):
    
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score


##########################
## Cutomized Objectives ##
##########################
#### Implement the method described in the paper:
# Ordinal Regression by Extended Binary Classification
# Ling Li, Hsuan-Tien Lin
def ebcObj(preds, dtrain):
    ## label are +1/-1
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    ## extended samples within the feature construction part
    if np.min(labels) == -1 and np.max(labels) == 1:
        s = np.exp(labels * preds)
        grad = - weights * labels / (1. + s)
        hess = weights * (labels**2) * s / ((1. + s)**2)
        ## TODO: figure out how to apply sample weights
    ## extended samples within the objective value computation part
    else:
        ## label are in [0,1,2,3]
        labels += 1
        M = preds.shape[0]
        N = preds.shape[1]
        grad = np.zeros((M,N), dtype=float)
        hess = np.zeros((M,N), dtype=float)
        ## we only use the first K-1 class for extended examples
        for c in range(N-1):
            k = c+1
            Y = 2. * np.asarray(labels > k, dtype=float) - 1.
            C_yk = np.power(Y - k, 2)
            C_yk1 = np.power(Y - (k+1), 2)
            w = np.abs(C_yk - C_yk1)
            p = preds[:,c]
            s = np.exp(Y * p)
            grad[:,c] = - w * Y / (1. + s)
            hess[:,c] = w * (Y**2) * s / ((1. + s)**2)
        ## apply sample weights
        grad *= weights[:,np.newaxis]
        hess *= weights[:,np.newaxis]
        grad.shape = (M*N)
        hess.shape = (M*N)
    return grad, hess

#### Implement the method described in the paper:
# Improving ranking performance with cost-sensitive ordinal classification via regression
# Yu-Xun Ruan, Hsuan-Tien Lin, and Ming-Feng Tsai
def cocrObj(preds, dtrain):
    ## label are in [0,1,2,3]
    Y = dtrain.get_label()
    Y = Y[:,np.newaxis]
    ## get sample weights
    weights = dtrain.get_weight()
    weights = weights[:,np.newaxis]
    ##
    M,N = preds.shape
    k = np.asarray(range(1,N+1))
    k = k[np.newaxis,:]
    b = np.asarray(Y >= k)
    C_yk = np.power(Y - k, 2)
    C_yk1 = np.power(Y - (k-1), 2)
    w = np.abs(C_yk - C_yk1)
    grad = 2 * w * (preds - b)
    hess = 2 * w

    ## apply sample weights
    grad *= weights
    hess *= weights
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess

#### directly optimized kappa (old version)
def softkappaObj(preds, dtrain):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]

    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j+1.))**2
        O += np.sum(wj * preds[:,j])
    
    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j+1.))**2) * preds[:,n] * (indicator - preds[:,j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:,n] = -M * (dO * E - O * dE) / (E**2)
        
        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j+1.))**2) * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,j])
       
        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:,n] = -M * ((d2O * E - O * d2E)*(E**2) - (dO * E - O * dE) * 2. * E * dE) / (E**4)

    grad *= -1.
    hess *= -1.
    # use a const
    #hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess) # It works!! no idea...
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess

#### directly optimized kappa (final version)
# since we use the cdf for finding cutoff which results in the same distribution between training/validaion
# so the denominator is kind of fixed
def softkappaObj(preds, dtrain, hess_scale=0.000125):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]
    weights = dtrain.get_weight()
    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j+1.))**2
        O += np.sum(wj * preds[:,j])
    
    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    #hist_pred = np.sum(preds, axis=0)
    hist_pred = hist_label
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j+1.))**2) * preds[:,n] * (indicator - preds[:,j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        #for k in range(N):
        #    for l in range(N):
        #        indicator = float(n == k)
        #        dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:,n] = -M * (dO * E - O * dE) / (E**2)
        
        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j+1.))**2) * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,j])
       
        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        #for k in range(N):
        #    for l in range(N):
        #        indicator = float(n == k)
        #        d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:,n] = -M * ((d2O * E - O * d2E)*(E**2) - (dO * E - O * dE) * 2. * E * dE) / (E**4)

    grad *= -1.
    hess *= -1.
    # use a const
    #hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = hess_scale / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess) # It works!! no idea...
    ## apply sample weights
    grad *= weights[:,np.newaxis]
    hess *= weights[:,np.newaxis]
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess
    

#####################
## Decoding Method ##
#####################
#### decoding method for ranking and regression
def output_function(x):
    if x<1:
        return 1
    elif x>8:
        return 8
    elif int(round(x))==3:
        return 2
    else:
        return int(round(x))
def getScore_1(pred):
    pred = list(pred)
    pred = [output_function(i)for i in pred]
    return pred
    

def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([8]*num, dtype=int)
    rank = pred.argsort()
    output[rank[:int(num*cdf[0]-1)]] = 1
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 2
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 3
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 4
    output[rank[int(num*cdf[3]):int(num*cdf[4]-1)]] = 5
    output[rank[int(num*cdf[4]):int(num*cdf[5]-1)]] = 6
    output[rank[int(num*cdf[5]):int(num*cdf[6]-1)]] = 7
    return output

#### get test score using cutoff found in the validation set
def getTestScore(pred, cutoff):
    num = pred.shape[0]
    output = np.asarray([4]*num, dtype=int)
    for i in range(num):
        if pred[i] <= cutoff[0]:
            output[i] = 1
        elif pred[i] <= cutoff[1]:
            output[i] = 2
        elif pred[i] <= cutoff[2]:
            output[i] = 3
    return output

#### decoding method for four class probabilities (e.g., softmax classification)
def getClfScore(preds, cdf):
    w = np.asarray(np.arange(1,config.n_classes+1))
    preds = preds * w[np.newaxis,:]
    preds = np.sum(preds, axis=1)
    output = getScore(preds, cdf)
    output = np.asarray(output, dtype=int)
    return output

#### decoding method for EBC
def applyEBCRule(preds, hard_threshold=False):
    if len(preds.shape) == 1:
        ## get prediction
        numOfSample = len(preds)/(config.n_classes-1)
        if hard_threshold:
            r = np.ones((numOfSample), dtype=int)
        else:
            print 
            r = np.ones((numOfSample), dtype=float)
        for c in range(config.n_classes-1):
            if hard_threshold:
                r += np.asarray(preds[c*numOfSample:(c+1)*numOfSample] > 0, dtype=int)
            else:
                r += preds[c*numOfSample:(c+1)*numOfSample]
    elif preds.shape[1] == 8:
        if hard_threshold:
            r = np.sum(np.asarray(preds[:,:7]>0), axis=1) + 1
        else:
            r = np.sum(preds[:,:7], axis=1) + 1
    return r

#### decoding method for COCR
def applyCOCRRule(preds):
    r = np.sum(preds, axis=1) + 1
    return r


#############################################
## Cutomized Evaluation Metric for XGBoost ##
#############################################
#### evalerror for regression and pairwise ranking
def evalerror_regrank_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() 
    preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

#### evalerror for softmax
def evalerror_softmax_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    preds = getClfScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

#### evalerror for soft-kappa
def evalerror_softkappa_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() 
    preds = softmax(preds)
    preds = getClfScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

#### evalerror for EBC
def evalerror_ebc_cdf(preds, dtrain, cdf, hard_threshold=False):
    labels = dtrain.get_label()
    ## extended samples within the feature construction part
    if np.min(labels) == -1 and np.max(labels) == 1:
        labels = applyEBCRule(labels)
    ## extended samples within the objective value computation part
    ## See ebcobj function for detail
    else:
        ## label are in [0,1,2,3]
        labels = labels
    #print preds.shape
    ## get prediction
    #hard = False
    if hard_threshold:
        preds = applyEBCRule(preds, hard_threshold=hard_threshold)
    else:
        preds = sigmoid(preds)
        preds = applyEBCRule(preds, hard_threshold=hard_threshold)
        preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

#### evalerror for COCR
def evalerror_cocr_cdf(preds, dtrain, cdf):
    labels = dtrain.get_label() 
    #print preds.shape
    ## get prediction
    #preds = sigmoid(preds)
    preds = applyCOCRRule(preds)
    preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

def minimize_quadratic_weighted_kappa(cutpoints,y_pred=None,y=None):
    cutpoints = np.sort(cutpoints)
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    y_pred = pd.cut(y_pred,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    score = quadratic_weighted_kappa(y,y_pred)
    return -score
def xgb_regression_quadratic_weighted_kappa(preds,dtrain):
    labels = dtrain.get_label()
    cutpoints = [1.886638,3.303624,4.152756,4.825063,5.653934,6.236325,6.765184]  
    res = minimize(minimize_quadratic_weighted_kappa,cutpoints,(preds,labels),method='BFGS')
    cutpoints = np.sort(res.x)
    cutpoints = np.concatenate([[-99999999999999999],cutpoints,[999999999999999]])
    y_pred = pd.cut(preds,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])
    kappa = quadratic_weighted_kappa(labels,y_pred)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)

"""
class WoE(BaseEstimator,TransformerMixin):
    #in this model , I will replace the category feature to 
    def __init__(self):
        self.cate_features = None
        
        
    def fit(self,X,y=None):
      
        self.cate_features = X.columns
        self.len = list(self.cate_features)
        message = X
        message['Response'] = y
        count_message = np.zeros([self.len,2])
        weight_message= np.zeros([self.len,2000,2])
        run = 0
        for cate in self.cate_features:
            values  = np.unique(message[cate])
            for val in values:
                for cal in range(1,9):
                    len1 = message[cate][message['Response']==cal].shape[0]
                    len2 = message[cate][message['Response']==cal and message[cate]==val].shape[0]
                    count_message[run,0] = len2 + count_message[run,0]
                    count_message[run,1] = len1 - len2 + count_message[run,1]
                    weight_message[run,val,0] = len2 + weight_message[run,val,0]
                    weight_message[run,val,1] = len1 - len2 + weight_message[run,val,1]
            run + 1
        #get weight of all value 
        woe_weight = np.zeros(self.len,2000)
        #woe_weight = ln()
       
            
        
        
        
        
        
    def transform(self,X):
        
        
    def count_messages(self,X,y):
"""        

        
                 
                
                
            
            
    
    
    
    
    