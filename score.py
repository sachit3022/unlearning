import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import numpy as np

from typing import List,Optional


import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from scipy.special import softmax

from sklearn.base import ClassifierMixin
import sklearn



from config import Config
from network import MiaModel
from dataset import SimpleDataSet, TorchStratifiedShuffleSplit

DEVICE = Config.DEVICE.value



        

def simple_mia(X, y,attack_model:ClassifierMixin ,n_splits=10):
    """Computes cross-validation score of a membership inference attack.

    Args:
      X : array_like of shape (n,).
        objective function evaluated on n samples.
      y : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(y)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits)

    return model_selection.cross_val_score(
        attack_model, X, y, cv=cv, scoring="accuracy"
    )

"""
#replaced by torch_neural_mia
def accuracy(model: nn.Module,val_dataset: Dataset,y: Optional[np.ndarray]=None) -> float:
    '''This functions computes the accuracy of a torch model on a torch dataset 

    Args:
        model (nn.Module): the model should have a predict method
        val_dataset (_type_): _description_
        y (List[in], optional): _description_. Defaults to None.

    Returns:
        accuracy (float) : the accuracy of the model on the dataset
    '''
        
    model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=2048, shuffle=True,num_workers=2)
    total_correct,total_count = 0,0
    for inputs,targets in val_dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        accuracy = (model.predict(inputs) == targets).sum() 
        total_correct += accuracy
        total_count += len(targets)
    return total_correct / total_count

    
def neural_mia(X: np.ndarray, y: np.ndarray, attack_model: nn.Module,n_splits: int=10):
    '''Computes cross-validation score of a membership inference attack.

    Args:
      X : array_like of shape (n,).
        objective function evaluated on n samples.
      y: array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    '''

    unique_members = np.unique(y)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    cv =TorchStratifiedShuffleSplit(n_splits=n_splits)

    return model_selection.cross_val_score(
        attack_model, SimpleDataSet(X,y), y, cv=cv, scoring=accuracy
    )
"""

def torch_neural_mia(X: np.ndarray, y: np.ndarray, attack_model: nn.Module,n_splits: int=10):
    """Computes cross-validation score of a membership inference attack.

    Args:
      X : array_like of shape (n,).
        objective function evaluated on n samples.
      y: array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int"""
    
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits)
    mia_scores= []
    X = softmax(X,axis=1)
    norm_X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_dataset = SimpleDataSet(norm_X, y)
    for i, (train_index, test_index) in enumerate(cv.split(X_dataset, y)):
        attack_model.fit(X_dataset[train_index])
        mia_scores.append(attack_model.score(X_dataset[test_index]))

    return np.array(mia_scores)
        

def compute_mia_score(test: List[float], forget: List[float]):
    """Computes the MIA score from test and train either loss of data or final logits. """

    # Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
    np.random.seed(Config.SEED.value)
    np.random.shuffle(forget)
    forget = forget[:len(test)]
    samples_mia = np.concatenate((test, forget))#.reshape((-1, 1))
    labels_mia = [0] * len(test) + [1] * len(forget)
    #dont trust neural_mia code write some test cases to validate.
    
    #attack_model = MiaModel([10,7,4],2)
    #mia_scores = torch_neural_mia( samples_mia, labels_mia ,attack_model, n_splits=3)

    attack_model = GradientBoostingClassifier()
    mia_scores = simple_mia( samples_mia, labels_mia ,attack_model, n_splits=5)
    return mia_scores.mean()



if __name__ == "__main__":
    # Example usage
    n = 1000
    
    #X = np.random.rand(n)
    #members = np.random.randint(2, size=n)

    #generate 2D gaussian  centered at [-5.00,-5.00] with std 1.0 and label 0 and centered at [5.0,5.0] with std 1.0 and label 1
    X = np.vstack([np.random.randn(n,2) + np.array([-5.0,-5.0]),np.random.randn(n,2) + np.array([5.0,5.0])])
    y = np.hstack([np.zeros(n),np.ones(n)])

    #plot the data
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.savefig("data.png")

    simple_mia_attack_model = linear_model.LogisticRegression()
    neural_mia_attack_model = MiaModel([2,10,10,2],2)

    for mia_fun,attack_model in [(simple_mia,simple_mia_attack_model), (torch_neural_mia,neural_mia_attack_model)]:
        mia_scores = mia_fun(X, y,attack_model,n_splits=2)
        #print(mia_scores)
        print(f"The {mia_fun.__name__} attack has an accuracy of {mia_scores.mean():.3f} on forgotten vs seen images")