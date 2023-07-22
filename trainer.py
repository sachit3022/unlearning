from typing import List, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os
import operator
from dataclasses import dataclass

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset


from config import dotdict
from plots import plot_losses
from dataset import TrainerDataLoaders

import logging
from torch.utils.tensorboard import SummaryWriter


# to track all the losses and other parameters. start with loss and accuracy.
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def count_parameters(model):
    # copied from pytorch disccussion forum
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer_and_scheduler(model, optimizer_config, scheduler_config):
    optimizer = operator.attrgetter(optimizer_config.type)(optim)(
        model.parameters(), **{k: v for k, v in optimizer_config.__dict__.items() if k!="type"})
    scheduler = operator.attrgetter(scheduler_config.type)(
        optim)(optimizer, **{k: v for k, v in scheduler_config.__dict__.items() if k!="type"})
    return optimizer, scheduler


@dataclass
class OptimizerConfig:
    type: str = "SGD"
    lr: float =  0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

@dataclass
class SchedulerConfig:
    type: str = "lr_scheduler.CosineAnnealingWarmRestarts"
    T_0:int =  10
    eta_min: float  =  1e-6
    last_epoch: int =  -1

@dataclass
class TrainerSettings:

    name: str = "trainer"
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    checkpoint: Optional[str] = None
    device: str = "cpu"
    verbose: bool = True
    log_freq: int = 1
    log_path: str = "logs"
    model_dir: str = "models"



    
class Trainer:
    def __init__(self,model: nn.Module, dataloaders: TrainerDataLoaders,trainer_args: TrainerSettings):


        self.name = trainer_args.name
        self.device = trainer_args.device

        self.model = model.to(self.device)

        self.train_loader = dataloaders.train
        self.val_loader = dataloaders.val
        self.test_loader = dataloaders.test

        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.model, trainer_args.optimizer, trainer_args.scheduler)
        self.loss_fn = trainer_args.loss_fn
        
        self.verbose = trainer_args.verbose
        self.init_trainer_tracker()

        #logging
        if self.verbose: 
            self.log_path = trainer_args.log_path
            self.writer = SummaryWriter(self.log_path + f"/runs/{self.name}")
            self.logger = logging.getLogger()
            self.log_freq = trainer_args.log_freq
            self.model_dir = trainer_args.model_dir
            
            
    @classmethod
    def load_from_checkpoint(cls, path,trainer_args):

        checkpoint = torch.load(path, map_location=cls.device)
        trainer = cls(trainer_args)
        trainer.load(checkpoint)
        return trainer

    def train(self, epochs: int):

        for epoch in tqdm(range(self.epoch+1, epochs+self.epoch+1), disable=not self.verbose):
            self.epoch = epoch
            self.train_epoch()
            self.test_epoch()
            self.log(epoch, {"train_loss": self.train_loss.avg, "test_loss": self.test_loss.avg, "train_accuracy": self.train_accuracy.avg,
                             "test_accuracy": self.test_accuracy.avg, "lr": self.scheduler.get_last_lr()[0],
                             "val_loss": self.val_loss.avg, "val_accuracy": self.val_accuracy.avg
                             })
            if self.test_accuracy.avg >= self.best_accuracy:
                self.best_accuracy = self.test_accuracy.avg
                self.best_loss = self.test_loss.avg
                self.best_model = self.model.state_dict()

        self.model.load_state_dict(self.best_model)
        if self.verbose : self.save(os.path.join(self.model_dir,f"weigths_{self.name}.pt")) 
        return self.model

    def train_epoch(self):
        self.model.train()
        self.train_loss.reset()
        self.train_accuracy.reset()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            self.train_loss.update(loss.item(), inputs.size(0))
            self.train_accuracy.update(self.accuracy(
                outputs, targets), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            if self.verbose:
                self.log_gradients()
            self.optimizer.step()
        self.scheduler.step()

    def test_epoch(self, debug=False):
        self.test(self.val_loader, self.val_loss, self.val_accuracy)
        if self.test_loader is not None:
            self.test(self.test_loader, self.test_loss, self.test_accuracy)
        self.debug_epoch(debug_id=self.epoch)

    @torch.no_grad()
    def test(self, loader: DataLoader, test_loss: AverageMeter, test_accuracy: AverageMeter):
        self.model.eval()
        test_loss.reset()
        test_accuracy.reset()
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            test_loss.update(loss.item(), inputs.size(0))
            test_accuracy.update(self.accuracy(
                outputs, targets), inputs.size(0))

    def load(self, path):

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.history = checkpoint['history']
        return self.model

    def _call_if_verbose(func):
        def wrapper(self, *args, **kwargs):
            if self.verbose and self.epoch % self.log_freq == 0:
                return func(self, *args, **kwargs)
            else:
                return None
        return wrapper

    @_call_if_verbose
    def save(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.best_model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'history': self.history
        }, path)


    @_call_if_verbose
    def log(self, epoch, logs):
        def group(x): return x.split("_")[-1]
        for key, value in logs.items():
            self.history[key].append(value)
            if self.verbose:
                self.writer.add_scalar(f"{group(key)}/{key}", value, epoch)

    @_call_if_verbose
    def log_gradients(self):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"{name}.grad", param.grad, self.epoch)

    @_call_if_verbose
    def debug_epoch(self, debug_id=0):
        """"Modify this function to log whatever you want to log in a debug epoch
        few things you can log:
        1. incorrect predictions
        3. activations
        4. weights
        5. train,test,valid datasnapshots
        6. MIA related stuff
        """

        self.model.eval()
        debug_data = defaultdict(list)

        # log correct and incorrect predictions with images and targets

        # debug_predictions = torch.arange(inputs.size(0))
        # debug_predictions =  (torch.argmax(probs,dim=1) != targets).nonzero().squeeze().detach() +batch_idx # only if we want to log incorrect predictions
        # debug_data["incorrect_predictions"].append((batch_idx*inputs.size(0) + debug_predictions).detach())
        # debug_data["targets"].append(targets[debug_predictions].detach())
        # debug_data["probs"].append(probs[debug_predictions].detach())
        """
        log_path = os.path.join(Config.LOG_PATH,self.name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({ name : torch.cat(each_tensors,dim=0) for name,each_tensors in debug_data.items()}, os.path.join(log_path, f"debug_predictions_{debug_id}.pt"))
        """
        if self.test_loader is not None:
            #rewrite this one
            criterion = nn.CrossEntropyLoss(reduction="none")
            all_losses, all_confidence = [], []
            all_loaders = [self.train_loader,self.val_loader]
            if self.test_loader is not None: all_loaders+=[self.test_loader] 
            for loader in  all_loaders:
                inputs, targets = next(iter(loader))
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                all_losses.append(criterion(logits, targets))
                all_confidence.append(torch.softmax(logits, dim=1).max(dim=1)[0])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            plot_losses(ax1, all_losses[1].flatten().detach().cpu(), all_losses[2].flatten(
            ).detach().cpu(), all_losses[0].flatten().detach().cpu())
            plot_losses(ax2, all_confidence[1].flatten().detach().cpu(), all_confidence[2].flatten(
            ).detach().cpu(), all_confidence[0].flatten().detach().cpu(), name="Confidence")    
            fig.savefig(os.path.join(self.log_path,f"losses_{self.epoch}.png"))
        return

    def accuracy(self, outputs, targets):

        _, preds = torch.max(outputs, dim=1)
        return torch.sum(preds == targets).item() / len(preds)

    def cross_validation_score(self, epochs=100):
        self.train(epochs)
        self.test_epoch()
        return self.val_accuracy.avg
    
    def reset(self):
        
        
        #reset model weights
        self.model.init_weights()
        #reset optimizer and scheduler
        self.optimizer.state = defaultdict(dict)
        self.scheduler.state = defaultdict(dict) 
        self.init_trainer_tracker()
    

    def init_trainer_tracker(self):
        self.train_loss, self.train_accuracy = AverageMeter(), AverageMeter()
        self.test_loss, self.test_accuracy = AverageMeter(), AverageMeter()
        self.val_loss, self.val_accuracy = AverageMeter(), AverageMeter()

        self.epoch = -1
        self.best_accuracy = 0
        self.best_loss = np.inf
        self.history = defaultdict(list)