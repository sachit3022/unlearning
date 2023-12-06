import os
import requests
from copy import deepcopy
from typing import Callable
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn import linear_model, model_selection
from sklearn.metrics import make_scorer, accuracy_score

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from datasets import get_cifar10_dataloaders, get_celeba_dataloaders
from torchvision.models import resnet18
import dotmap
from torch.nn import functional as tr_f
import random


DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())


def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for batch in loader:
        inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total

def compute_outputs(net, loader):
    """Auxiliary function to compute the logits for all datapoints.
    Does not shuffle the data, regardless of the loader.
    """

    # Make sure loader does not shuffle the data
    if isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler):
        loader = DataLoader(
            loader.dataset, 
            batch_size=loader.batch_size, 
            shuffle=False, 
            num_workers=loader.num_workers)
    
    all_outputs = []
    for batch in loader:
        inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)

        logits = tr_f.softmax(net(inputs), dim=-1).detach().cpu().numpy() # (batch_size, num_classes)
        
        all_outputs.append(logits)
        
    return np.concatenate(all_outputs) # (len(loader.dataset), num_classes)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false positive rate (FPR)."""
    fp = np.sum(np.logical_and((y_pred == 1), (y_true == 0)))
    n = np.sum(y_true == 0)
    return fp / n


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false negative rate (FNR)."""
    fn = np.sum(np.logical_and((y_pred == 0), (y_true == 1)))
    p = np.sum(y_true == 1)
    return fn / p

# The SCORING dictionary is used by sklearn's `cross_validate` function so that
# we record the FPR and FNR metrics of interest when doing cross validation
SCORING = {
    'false_positive_rate': make_scorer(false_positive_rate),
    'false_negative_rate': make_scorer(false_negative_rate),
    'accuracy': make_scorer(accuracy_score)
}

def cross_entropy_f(x):
    # To ensure this function doesn't fail due to nans, find
    # all-nan rows in x and substitude them with all-zeros.
    x[np.all(np.isnan(x), axis=-1)] = np.zeros(x.shape[-1])
    
    pred = torch.tensor(np.nanargmax(x, axis = -1))
    x = torch.tensor(x)

    fn = nn.CrossEntropyLoss(reduction="none")
    
    return fn(x, pred).numpy()


def logistic_regression_attack(
        outputs_U, outputs_R, n_splits=2, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      fpr, fnr : float * float
    """
    assert len(outputs_U) == len(outputs_R)
    samples = np.concatenate((outputs_R, outputs_U)).reshape((-1, 1))
    
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    scores =  model_selection.cross_validate(
        attack_model, samples, labels, cv=cv, scoring=SCORING)
    
    fpr = np.mean(scores["test_false_positive_rate"])
    fnr = np.mean(scores["test_false_negative_rate"])
    return fpr, fnr

def best_threshold_attack(
        outputs_U: np.ndarray, 
        outputs_R: np.ndarray, 
        random_state: int = 0
    ) -> tuple[list[float], list[float]]:
    """Computes FPRs and FNRs for an attack that simply splits into 
    predicted positives and predited negatives based on any possible 
    single threshold.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
    Returns:
      fpr, fnr : list[float] * list[float]
    """
    assert len(outputs_U) == len(outputs_R)
    
    samples = np.concatenate((outputs_R, outputs_U))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    N = len(outputs_U)
    
    fprs, fnrs = [], []
    for thresh in sorted(list(samples.squeeze())):
        ypred = (samples > thresh).astype("int")
        fprs.append(false_positive_rate(labels, ypred))
        fnrs.append(false_negative_rate(labels, ypred))
    

    return fprs, fnrs

def compute_epsilon_s(fpr: list[float], fnr: list[float], delta: float) -> float:
    """Computes the privacy degree (epsilon) of a particular forget set example, 
    given the FPRs and FNRs resulting from various attacks.
    
    The smaller epsilon is, the better the unlearning is.
    
    Args:
      fpr: list[float] of length m = num attacks. The FPRs for a particular example. 
      fnr: list[float] of length m = num attacks. The FNRs for a particular example.
      delta: float
    Returns:
      epsilon: float corresponding to the privacy degree of the particular example.
    """
    assert len(fpr) == len(fnr)
    
    per_attack_epsilon = [0.]
    for fpr_i, fnr_i in zip(fpr, fnr):
        #print(fpr_i,fnr_i)
        if fpr_i == 0 and fnr_i == 0:
            per_attack_epsilon.append(np.inf)
        elif fpr_i == 0 or fnr_i == 0:
            pass # discard attack
        else:
            with np.errstate(invalid='ignore'):
                epsilon1 = np.log(1. - delta - fpr_i) - np.log(fnr_i)
                epsilon2 = np.log(1. - delta - fnr_i) - np.log(fpr_i)
            if np.isnan(epsilon1) and np.isnan(epsilon2):
                per_attack_epsilon.append(np.inf)
            else:
                per_attack_epsilon.append(np.nanmax([epsilon1, epsilon2]))
    
    return np.nanmax(per_attack_epsilon)

def bin_index_fn(
        epsilons: np.ndarray, 
        bin_width: float = 0.5, 
        B: int = 13
        ) -> np.ndarray:
    """The bin index function."""
    bins = np.arange(0, B) * bin_width
    return np.digitize(epsilons, bins)


def F(epsilons: np.ndarray) -> float:
    """Computes the forgetting quality given the privacy degrees 
    of the forget set examples.
    """
    #print(epsilons)
    ns = bin_index_fn(epsilons)
    hs = 2. / 2 ** ns
    return np.mean(hs)


def forgetting_quality(
        outputs_U: np.ndarray, # (N, S)
        outputs_R: np.ndarray, # (N, S)
        attacks: list[Callable] = [logistic_regression_attack],
        delta: float = 0.01
    ):
    """
    Both `outputs_U` and `outputs_R` are of numpy arrays of ndim 2:
    * 1st dimension coresponds to the number of samples obtained from the 
      distribution of each model (N=512 in the case of the competition's leaderboard) 
    * 2nd dimension corresponds to the number of samples in the forget set (S).
    """
    
    # N = number of model samples
    # S = number of forget samples
    N, S = outputs_U.shape
    
    assert outputs_U.shape == outputs_R.shape, \
        "unlearn and retrain outputs need to be of the same shape"
    
    epsilons = []
    pbar = tqdm(range(S))
    for sample_id in pbar:
        pbar.set_description("Computing F...")
        
        sample_fprs, sample_fnrs = [], []
        
        for attack in attacks: 
            uls = outputs_U[:, sample_id]
            rls = outputs_R[:, sample_id]
            
            fpr, fnr = attack(uls, rls)
            
            if isinstance(fpr, list):
                sample_fprs.extend(fpr)
                sample_fnrs.extend(fnr)
            else:
                sample_fprs.append(fpr)
                sample_fnrs.append(fnr)        
        sample_epsilon = compute_epsilon_s(sample_fprs, sample_fnrs, delta=delta)
        epsilons.append(sample_epsilon)
        
    return F(np.array(epsilons))

def score_unlearning_algorithm(
        data_loaders: dict, 
        pretrained_models_path: list[str],
        unlearning_models_path: list[str], 
        base_model: nn.Module,
        dataset : str,
        n: int = 10,
        delta: float = 0.01,
        f: Callable = cross_entropy_f,
        attacks: list[Callable] = [logistic_regression_attack,best_threshold_attack] #best_threshold_attack,
        ) -> dict:
    
    # n=512 in the case of unlearn and n=1 in the
    # case of retrain, since we are only provided with one retrained model here

    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    val_loader = data_loaders["validation"]
    test_loader = data_loaders["testing"]
    model = deepcopy(base_model).to(DEVICE)

    outputs_U = []
    retain_accuracy_U = []
    test_accuracy_U = []
    forget_accuracy_U = []

    outputs_R = []
    retain_accuracy_R = []
    test_accuracy_R = []
    forget_accuracy_R = []


    for model_path in tqdm(unlearning_models_path):
        if "model_state_dict" in torch.load(model_path,map_location=DEVICE).keys():
            model.load_state_dict(torch.load(model_path,map_location=DEVICE)["model_state_dict"])
        else:
            model.load_state_dict(torch.load(model_path,map_location=DEVICE))

        outputs_Ui = compute_outputs(model, forget_loader) 
        # The shape of outputs_Ui is (len(forget_loader.dataset), 10)
        # which for every datapoint is being cast to a scalar using the funtion f
        outputs_U.append( f(outputs_Ui) )
        retain_accuracy_U.append(accuracy(model, retain_loader))
        test_accuracy_U.append(accuracy(model, test_loader))
        forget_accuracy_U.append(accuracy(model, forget_loader))
    
    for model_path in tqdm(pretrained_models_path):
        if "model_state_dict" in torch.load(model_path,map_location=DEVICE).keys():
            model.load_state_dict(torch.load(model_path,map_location=DEVICE)["model_state_dict"])
        else:
            model.load_state_dict(torch.load(model_path,map_location=DEVICE)["model_state_dict"])
        outputs_Ri = compute_outputs(model, forget_loader) 
        outputs_R.append( f(outputs_Ri) )
        retain_accuracy_R.append(accuracy(model, retain_loader))
        test_accuracy_R.append(accuracy(model, test_loader))
        forget_accuracy_R.append(accuracy(model, forget_loader))

    outputs_R = np.array(outputs_R)
    outputs_U = np.array(outputs_U)
        

    RAR = np.mean(retain_accuracy_R)
    TAR = np.mean(test_accuracy_R)
    FAR = np.mean(forget_accuracy_R)

    RAU = np.mean(retain_accuracy_U)
    TAU = np.mean(test_accuracy_U)
    FAU = np.mean(forget_accuracy_U)

    RA_ratio = RAU / RAR
    TA_ratio = TAU / TAR

    np.save("outputs_U.npy", outputs_U)
    np.save("outputs_R.npy", outputs_R)

    f = forgetting_quality(
    outputs_U, 
    outputs_R,
    attacks=attacks,
    delta=delta)

    return {
        "total_score": f * RA_ratio * TA_ratio,
        "F": f,
        "unlearn_retain_accuracy": RAU,
        "unlearn_test_accuracy": TAU, 
        "unlearn_forget_accuracy": FAU,
        "retrain_retain_accuracy": RAR,
        "retrain_test_accuracy": TAR, 
        "retrain_forget_accuracy": FAR,

    }


def test( n: int = 10,
        delta: float = 0.01,
        f: Callable = cross_entropy_f,
        attacks: list[Callable] = [best_threshold_attack, logistic_regression_attack]):
    outputs_U = np.load("outputs_U.npy")
    outputs_R  = np.load("outputs_R.npy")

    f = forgetting_quality(
    outputs_U, 
    outputs_U,
    attacks=[logistic_regression_attack],
    delta=delta)
    return f

def plot():
    outputs_U = np.load("outputs_U.npy")
    outputs_R  = np.load("outputs_R.npy")
    plt.hist(outputs_U.mean(axis=0),bins=100)
    plt.hist(outputs_R.mean(axis=0),bins=100)
    plt.legend(["Unlearn","Retrain"])
    plt.savefig("hist.png")








if __name__ == "__main__":
    #print(test())
    #plot()

    
    dataset = "celeba" #"cifar10"
    method = "sachit" #"prune

    unlearnt_model_path = f"neurips-submission/{method}_unlearn_{dataset}" #"neurips-submission/finetune_unlearn_cifar10"
    retrain_model_path = "neurips-submission/retrain_celeba" if dataset == "celeba" else "neurips-submission/retrain_cifar10"
    num_classes = 8 if dataset == "celeba" else 10
 
    args = dotmap.DotMap({"data":dotmap.DotMap({"BATCH_SIZE":512,"num_classes":num_classes,"num_workers":6}), "directory":dotmap.DotMap({"LOG_PATH":"./logs/"}),"device":DEVICE})
    
    DEVICE  = args.device
    train_loader,retain_loader, forget_loader, validation_loader,test_loader = get_celeba_dataloaders(args,balanced=False) if dataset == "celeba" else get_cifar10_dataloaders(args,balanced=False)
    dataloaders = {"retain":retain_loader,"forget":forget_loader,"validation":validation_loader,"testing":test_loader}
    
    unlearn_models = [os.path.join(unlearnt_model_path,model) for model in np.random.choice(os.listdir(unlearnt_model_path),32,replace=True)]
    retrain_models = [os.path.join(retrain_model_path,model) for model in np.random.choice(os.listdir(retrain_model_path),32,replace=False)]
    base_model =  resnet18(num_classes=num_classes).to(DEVICE)
    score = score_unlearning_algorithm(dataloaders,retrain_models,unlearn_models,base_model,dataset)
    print(score)
    