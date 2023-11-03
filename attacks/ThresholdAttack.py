from .Attack import Attack
import numpy as np
from scipy.special import softmax

class ThresholdAttack(Attack):
    def __init__(self,threshold):
        self.threshold = threshold
        self.model = np.vectorize(lambda x: x>self.threshold)
        super().__init__(self.model)

    def __call__(self,og,ul):
        x1,x2 = softmax(og,axis=1),softmax(ul,axis=1)
        y_hat = self.model(x1[:,0])
        y = self.model(x2[:,0])
        self.fpr = np.sum(y_hat[y==0])/np.sum(y==0)
        self.fnr = np.sum(1-y_hat[y==1])/np.sum(y==1)
        return self.fpr,self.fnr
    
        


        
        
