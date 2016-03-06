import numpy as np
def gradient(y, pred,gamma=0.9):
    pred = pred.ravel()
    diff = pred-y
    g = diff/np.sqrt(gamma**2+diff**2)
    return g

def hessian(y, pred,gamma=0.9):
    pred = pred.ravel()
    diff = y - pred
    tmp = (gamma**2+diff**2)
    h = gamma**2*(np.power(tmp,-3/2.0))
    return h

def pseudo_huber(pred,dtrain,gamma=0.9):
    y = dtrain.get_label()
    g = gradient(y,pred,gamma=gamma)
    h = hessian(y,pred,gamma=gamma)
    return g,h