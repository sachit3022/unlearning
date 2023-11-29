from .Attack    import Attack
import numpy as np
from scipy.special import softmax
from matplotlib import pyplot as plt


#copied from stackoverflow
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return sorted(np.roots([a,b,c]))


class GaussianAttack(Attack):
    def __init__(self,index = 0):
        self.index = index
        self.threshold = [0.5,0.5]
        self.model = np.vectorize(lambda x: x>self.threshold[0] and x<self.threshold[1])
        super().__init__(self.model)
    def fit(self,X,y):
        mean_og,std_og = np.mean(X[y==0]),np.std(X[y==0])
        mean_ul,std_ul = np.mean(X[y==1]),np.std(X[y==1])
        result = solve(mean_og,mean_ul,std_og,std_ul)
        self.threshold = result
        
        #check delta of result
        gaussain_og = lambda x: np.exp(-0.5*((x-mean_og)/std_og)**2)/(std_og*np.sqrt(2*np.pi))
        gaussain_ul = lambda x: np.exp(-0.5*((x-mean_ul)/std_ul)**2)/(std_ul*np.sqrt(2*np.pi))

        a = gaussain_og(result[0]+0.01)
        b = gaussain_ul(result[0]+0.01)
        if a>b:
            self.model = np.vectorize(lambda x: x<=self.threshold[0] or x<=self.threshold[1])
        else:
            self.model = np.vectorize(lambda x: x>=self.threshold[0] and x<=self.threshold[1])

       
    def __call__(self,og,ul):
       
        X = softmax(np.vstack([og,ul]),axis=1)[:,self.index]
        y = np.vstack([np.ones_like(og),np.zeros_like(ul)])[:,self.index]
        self.fit(X,y)        
        y_hat = self.model(X)
        self.fpr = np.sum(y_hat[y==0])/np.sum(y==0)
        self.fnr = np.sum(1-y_hat[y==1])/np.sum(y==1)
        return self.fpr,self.fnr
    