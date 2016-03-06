import numpy as np
from sklearn.base import BaseEstimator
from minepy import MINE

class SelectMic(BaseEstimator):

    def __init__(self,beta=0.1,n=None,verbose=0):
        # the balance between maximizing relevance and minimizing redundancy is determined by parameter beta
        self.beta = beta
        # number of features to select
        self.n = n
        self.verbose = verbose
    def fit(self,X,y):
        # initialize phi and feature set
        # if number of features is not set, half of the features will be selected
        n = self.n
        beta = self.beta
        verbose = self.verbose
        if n ==None:
            n = int(X.shape[0]/2)

        features = np.arange(X.shape[1]).tolist()
        best_mi = -np.inf
        X_hat = 0
        for xi in features:
            m = MINE()
            m.compute_score(X[:,xi],y)
            #compute I(xi,y) and get max xi
            mi_xi_y = m.mic()
            if best_mi<mi_xi_y:
                X_hat = xi
        phi = [X_hat]
        features.remove(X_hat)
        # get paris for elements in phi and features
        while len(phi)<n:
            mi_scores = np.zeros(len(features))
            for xi_idx,xi in enumerate(features):
                m = MINE()
                m.compute_score(X[:,xi],y)
                #compute I(xi,y)
                mi_xi_y = m.mic()
                sum_mi_xi_xj = 0
                for xj in phi:
                    # compute I(xi,xj) and save for further evaluation
                    m = MINE()
                    m.compute_score(X[:,xi],X[:,xj])
                    mi_xi_xj = m.mic()
                    sum_mi_xi_xj+=mi_xi_xj
                mi_scores[xi_idx] = mi_xi_y - beta*sum_mi_xi_xj
                if verbose>=2:
                    print "mi_scores for xi:{xi}, xj:{xj} is {mi_scores}".format(xi=xi,xj=xj,mi_scores=mi_scores[xi_idx])

            X_hat = np.argmax(mi_scores)
            if verbose==1:
                print "X_hat is {X_hat}".format(X_hat=X_hat)
            X_hat = features[X_hat]
            phi.append(X_hat)
            features.remove(X_hat)
        self.phi = phi
        self.features = features
        
    def transform(self,X):
        phi = self.phi
        return X[:,phi]