from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset

class TorchStratifiedShuffleSplit:
    def __init__(self,n_splits:int=5,random_state:int=0):
        self.n_splits = n_splits
        self.random_state = random_state
    def get_n_splits(self, X):
        return self.n_splits

    def split(self, dataset,y=None,groups=None):
        cv = StratifiedShuffleSplit(n_splits=self.n_splits,random_state = self.random_state)
        args  =    zip(*[(i,y) for i,(X,y) in enumerate(dataset)])
        for train_idx, val_idx in iter(cv.split(*args)):
            yield train_idx,val_idx
                                       
    
class SimpleDataSet(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.int64) 
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.X[index],self.y[index]