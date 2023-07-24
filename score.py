from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, model_selection
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


from network import MiaModel
from trainer import Trainer, TrainerSettings
from dataset import SimpleDataSet,create_dataloaders,TrainerDataLoaders
from config import dotdict

import sys
from dataclasses import dataclass
score = sys.modules[__name__]

@dataclass
class MiaTrainerSettings(TrainerSettings):
    verbose: bool = False
    batch_size: int = 4096
    num_workers: int =  32
    epochs: int = 200


class MiaScore:
    # this is abstract function needs to be run without args. remove and move to trainer and args.
    CHAR_ORDER = {"loss": 0, "logits": 1, "gradients": 2}

    def __init__(self, parent_model, attack_model, charecterstic :str = "gradients", folds :int =5, attack_train_settings: MiaTrainerSettings = MiaTrainerSettings()):

        self.model = parent_model
        self.charecterstic = charecterstic
        self.attack_model = attack_model
        self.folds = folds
        self.device = next(self.attack_model.parameters()).device
        self.attack_train_args = attack_train_settings

    def compute_inputs_to_mia(self, loader):
        """Auxiliary function to compute per-sample losses"""

        all_features = []
        hook_features = []

        self.model.eval()
        hook_features = []

        def hook_fn(module, grad_input, grad_output): return hook_features.append(
            grad_input[0].flatten(1))
        handle = self.model.fc.register_full_backward_hook(hook_fn)

        criterion = nn.CrossEntropyLoss(reduction="none")
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            logits = self.model(inputs)
            losses = criterion(logits, targets)
            loss = losses.mean()
            loss.backward()
            all_features.append((losses.reshape((-1, 1)), logits))
            if batch_idx == 5:
                break
        handle.remove()
        self.model.zero_grad()
        return [torch.vstack(f).detach().cpu() for f in zip(*all_features)] + [torch.vstack(hook_features).detach().cpu()]

    def sklearn_mia(self, X, y):
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

        cv = model_selection.StratifiedShuffleSplit(n_splits=self.folds)

        return model_selection.cross_val_score(
            self.attack_model, X, y, cv=cv, scoring="accuracy"
        )

    def pytorch_mia(self, X, y):

        attack_dataset = SimpleDataSet(X, y)
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True)

        for train_index, test_index in skf.split(attack_dataset.X, attack_dataset.y):

            self.attack_model.to(self.device)
            
            dataloaders = {"train" : DataLoader(Subset(attack_dataset, train_index), pin_memory=True, persistent_workers=True,
                                               batch_size=self.attack_train_args.batch_size,num_workers=self.attack_train_args.num_workers),
                          "val" : DataLoader(Subset(attack_dataset, test_index), batch_size=self.attack_train_args.batch_size,
                                          shuffle=False, num_workers=self.attack_train_args.num_workers , pin_memory=True, persistent_workers=True)
                                          }
            
            #get optimizer and scheduler
            attack_trainer_settings = MiaTrainerSettings( optimizer=self.attack_train_args.optimizer, scheduler= self.attack_train_args.scheduler, device=self.device, **{k:v for k,v in self.attack_train_args.__dict__.items() if k not in {"optimizer","device","scheduler","train","epochs"}})
            trainer = Trainer(model = self.attack_model, dataloaders=TrainerDataLoaders(**dataloaders), trainer_args=attack_trainer_settings )
            trainer.reset()
            yield trainer.cross_validation_score(epochs=self.attack_train_args.epochs)


    def cross_validation_score(self, test_features, forget_features, folds=5):
        X = torch.cat((test_features, forget_features))
        y = torch.tensor([0] * len(test_features) + [1] *
                         len(forget_features), dtype=torch.int64)

        if isinstance(self.attack_model, ClassifierMixin):
            return self.sklearn_mia(X.numpy(), y.numpy())
        else:
            return self.pytorch_mia(X, y)

    def compute_model_mia_score(self, forget_loader, test_loader):

        test_feature_list = self.compute_inputs_to_mia(test_loader)
        forget_feature_list = self.compute_inputs_to_mia(forget_loader)

        test_features, forget_features = test_feature_list[self.CHAR_ORDER[self.charecterstic]
                                                           ], forget_feature_list[self.CHAR_ORDER[self.charecterstic]]
        balance_length = min(len(test_features), len(forget_features))

        mia_score = next(self.cross_validation_score(
            test_features[:balance_length], forget_features[:balance_length], folds=5))
        return mia_score