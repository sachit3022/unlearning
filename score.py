import sys
from dataclasses import dataclass
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
from dataset import SimpleDataSet,TrainerDataLoaders
import trainer as tr

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

    def compute_inputs_to_mia(self, loader,forget=False):
        """Auxiliary function to compute per-sample losses"""

        all_features = []
        hook_features = []

        self.model.eval()

        hook_features = []

        def hook_fn(module, grad_input, grad_output): return hook_features.append(
            grad_input[0].flatten(1))
        handle = self.model.fc.register_full_backward_hook(hook_fn)

        criterion = nn.CrossEntropyLoss(reduction="none")
        for batch_idx, batch_data in enumerate(loader):
            
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
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

        test_feature_list = self.compute_inputs_to_mia(test_loader,forget=False)
        forget_feature_list = self.compute_inputs_to_mia(forget_loader,forget=True)

        test_features, forget_features = test_feature_list[self.CHAR_ORDER[self.charecterstic]
                                                           ], forget_feature_list[self.CHAR_ORDER[self.charecterstic]]
        balance_length = min(len(test_features), len(forget_features))

        mia_score = next(self.cross_validation_score(
            test_features[:balance_length], forget_features[:balance_length], folds=5))
        return mia_score
    


class RetrainMIAScore:
    # this is abstract function needs to be run without args. remove and move to trainer and args.
    CHAR_ORDER = {"loss": 0, "logits": 1, "gradients": 2}

    def __init__(self, parent_model,retrain_model, attack_model, charecterstic :str = "gradients", folds :int =5, attack_train_settings: MiaTrainerSettings = MiaTrainerSettings()):

        self.model = parent_model
        self.scratch_model = retrain_model
        self.charecterstic = charecterstic
        self.attack_model = attack_model
        self.folds = folds
        self.device = next(self.attack_model.parameters()).device
        self.attack_train_args = attack_train_settings

    def compute_inputs_to_mia(self, loader,forget=False):
        """Auxiliary function to compute per-sample losses"""

        all_features = []
        hook_features = []

        hook_features = []

        if forget:
           model = self.model 
        else:
            model = self.scratch_model
        model.eval()

        def hook_fn(module, grad_input, grad_output): return hook_features.append(
            grad_input[0].flatten(1))
        handle = model.fc.register_full_backward_hook(hook_fn)

        criterion = nn.CrossEntropyLoss(reduction="none")
        for batch_idx, batch_data in enumerate(loader):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            model.zero_grad()
            logits = model(inputs)
            losses = criterion(logits, targets)
            loss = losses.mean()
            loss.backward()
            all_features.append((losses.reshape((-1, 1)), logits))
            if batch_idx == 5:
                break
        handle.remove()
        model.zero_grad()
        return [torch.vstack(f).detach().cpu() for f in zip(*all_features)] + [torch.vstack(hook_features).detach().cpu()]

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


    def cross_validation_score(self, test_features, forget_features):
        X = torch.cat((test_features, forget_features))
        y = torch.tensor([0] * len(test_features) + [1] *
                         len(forget_features), dtype=torch.int64)

        if isinstance(self.attack_model, ClassifierMixin):
            return self.sklearn_mia(X.numpy(), y.numpy())
        else:
            return self.pytorch_mia(X, y)

    def compute_model_mia_score(self, test_loader):

        test_feature_list = self.compute_inputs_to_mia(test_loader,forget=False)
        forget_feature_list = self.compute_inputs_to_mia(test_loader,forget=True)

        test_features, forget_features = test_feature_list[self.CHAR_ORDER[self.charecterstic]
                                                           ], forget_feature_list[self.CHAR_ORDER[self.charecterstic]]
        balance_length = min(len(test_features), len(forget_features))

        mia_score = next(self.cross_validation_score(
            test_features[:balance_length], forget_features[:balance_length]))
        return mia_score
    

def compute_unlearning_metrics(args, model, dataloaders):
    mia_scores = {"retain_forget":0,"forget_test":0,"retain_test":0}
    for name, t, f, in [("retain_forget", dataloaders.retain, dataloaders.forget), ("forget_test", dataloaders.forget,  dataloaders.test), ("retain_test", dataloaders.retain, dataloaders.test)]:
        attack_model = getattr(score, args.attack_model.type)(**args.attack_model.model_args).to(args.device)
        attack_optimizer_config = getattr(tr, args.attack_trainer.optimizer.type + "OptimConfig" )(**args.attack_trainer.optimizer)
        attack_scheduler_config = getattr(tr, args.attack_trainer.scheduler.type + "SchedulerConfig" )(**args.attack_trainer.scheduler)
        mia_args = MiaTrainerSettings(optimizer=attack_optimizer_config,num_classes=2,log_path=args.directory.LOG_PATH,model_dir=args.MODEL_DIR, scheduler=attack_scheduler_config,**{k:v for k,v in args.attack_trainer.items() if k not in {"optimizer","scheduler","train"}})
        miaScore = MiaScore(parent_model=model, attack_model=attack_model,charecterstic = args.attack_model.charecterstic,folds=args.attack_model.folds, attack_train_settings=mia_args)
        mia_scores[name] = miaScore.compute_model_mia_score(t, f)
    return mia_scores

def compute_retrain_unlearning_metrics(args, model,retrain_model,dataloaders):
    mia_scores = {"retain":0,"forget":0,"test":0}
    for name, t, in [("retain", dataloaders.retain), ("forget", dataloaders.forget), ("test", dataloaders.test)]:
        attack_model = getattr(score, args.attack_model.type)(**args.attack_model.model_args).to(args.device)
        attack_optimizer_config = getattr(tr, args.attack_trainer.optimizer.type + "OptimConfig" )(**args.attack_trainer.optimizer)
        attack_scheduler_config = getattr(tr, args.attack_trainer.scheduler.type + "SchedulerConfig" )(**args.attack_trainer.scheduler)
        mia_args = MiaTrainerSettings(optimizer=attack_optimizer_config,num_classes=2,log_path=args.directory.LOG_PATH,model_dir=args.MODEL_DIR, scheduler=attack_scheduler_config,**{k:v for k,v in args.attack_trainer.items() if k not in {"optimizer","scheduler","train"}})
        miaScore = RetrainMIAScore(parent_model=model,retrain_model=retrain_model, attack_model=attack_model,charecterstic = args.attack_model.charecterstic,folds=args.attack_model.folds, attack_train_settings=mia_args)
        mia_scores[name] = miaScore.compute_model_mia_score(t)
    return mia_scores