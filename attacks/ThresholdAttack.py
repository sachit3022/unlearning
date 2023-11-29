from .Attack import Attack
import numpy as np
from scipy.special import softmax

class ThresholdAttack(Attack):
    def __init__(self,threshold,index = 0):
        self.threshold = threshold
        self.index = index
        self.model = np.vectorize(lambda x: x>self.threshold)
        super().__init__(self.model)

    def __call__(self,og,ul):
        X = softmax(np.vstack([og,ul]),axis=1)
        y = np.vstack([np.ones_like(og),np.zeros_like(ul)])[:,self.index]
        y_hat = self.model(X[:,self.index])
        self.fpr = np.sum(y_hat[y==0])/np.sum(y==0)
        self.fnr = np.sum(1-y_hat[y==1])/np.sum(y==1)
        return self.fpr,self.fnr
    



        
        
