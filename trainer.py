from _collections_abc import dict_items
from typing import Any, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os
import operator
from dataclasses import dataclass
import random
from enum import Enum

import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


import logging
import sys
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import  ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score

import time

from plots import plot_losses,plot_image_grid,plot_confusion_matrix,GradCamWrapper,LossSurface
from datasets import TrainerDataLoaders, UnlearnCelebADataset



from torchvision.models import resnet18




tr = sys.modules[__name__]

# to track all the losses and other parameters. start with loss and accuracy.
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
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
        tr)(optimizer, **{k: v for k, v in scheduler_config.__dict__.items() if k!="type"})
    return optimizer, scheduler

@dataclass
class OptimConfig:
    type : str =  "SGD"
    lr: float = 0.1

@dataclass
class AdamWOptimConfig(OptimConfig):
    type: str = "AdamW"
    lr: float = 1e-5
    weight_decay: float = 1e-2
    betas: Tuple[float] = (0.9, 0.999)

@dataclass
class SGDOptimConfig(OptimConfig):
    type: str = "SGD"
    lr: float =  0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

@dataclass
class SchedulerConfig:
    type: str = "CosineAnnealingLR"

@dataclass
class CosineAnnealingLRSchedulerConfig(SchedulerConfig):
    type: str = "CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 0


@dataclass
class TrainerSettings:

    name: str = "trainer"
    optimizer: OptimConfig = SGDOptimConfig()
    scheduler: OptimConfig = CosineAnnealingLRSchedulerConfig()
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    checkpoint: Optional[str] = None
    device: str = "cpu"
    verbose: bool = True
    log_freq: int = 1
    log_path: str = "logs"
    model_dir: str = "models"
    num_classes: int = 8


#only valid for num classes 2 if more just an average works
class PrecesionFRR(AverageMeter):
    def __init__(self,num_classes=2) -> None:
        self.fpr = 0.1
    def update(self,inputs,outputs):
        start_theshold,end_theshold = 0,1
        for _ in range(10):

            mid = (start_theshold+end_theshold)/2.000000
            if self.compute_fpr(inputs,outputs,mid) < self.fpr:
                end_theshold = mid
            else:
                start_theshold = mid
        
        #now we have the theshold calculate the precision and recall
        precision = self.compute_precision(inputs,outputs,mid)
        print(mid,self.compute_fpr(inputs,outputs,mid),precision)
        self.sum += precision * len(inputs)
        self.count += len(inputs)
        self.avg = self.sum / self.count

    def compute_fpr(self,inputs,outputs,theshold):
        FP = torch.logical_and(inputs<theshold, outputs==1).sum()
        TN = torch.logical_and(inputs<theshold, outputs==0).sum()
        if FP+TN == 0: return 0
        return FP/(FP+TN)
    def compute_precision(self,inputs,outputs,theshold):
        FP = torch.logical_and(inputs<theshold, outputs==1).sum()
        TP = torch.logical_and(inputs<theshold,outputs==0).sum()
        return (TP/(FP+TP)).item()


        


class SingleConfusionMatrix:
    def __init__(self,task='binary',num_classes=2) -> None:
        self.cm = ConfusionMatrix(task=task,num_classes=num_classes)
    def update(self,inputs,outputs):
        self.cm.update(inputs,outputs)
    def compute(self):
        self.cm.compute()
    def reset(self):
        self.cm.reset()
    def plot(self,filename):
        
        plot_confusion_matrix(self.cm.compute().cpu().numpy(),filename=f"{filename.split('.')[0]}.{filename.split('.')[1]}")
    def to(self,device):
        self.cm.to(device)
        return self

class MultiConfusionMatrix:
    def __init__(self,task='binary',num_heads=40,num_classes=2) -> None:
        self.cms = [ConfusionMatrix(task=task,num_classes=num_classes) for _ in range(num_heads)]
    def update(self,inputs,outputs):
        #format BXH and H=40
        for i in range(len(self.cms)):
            self.cms[i].update(inputs[:,i],outputs[:,i])
    def compute(self):
        for i in range(len(self.cms)):
            self.cms[i].compute()
    def reset(self):
        for i in range(len(self.cms)):
            self.cms[i].reset()
    def plot(self,filename):
        for i in range(len(self.cms)):
            plot_confusion_matrix(self.cms[i].compute().cpu().numpy(),filename=f"{filename.split('.')[0]}_{i}.{filename.split('.')[1]}")
    def to(self,device):
        for i in range(len(self.cms)):
            self.cms[i].to(device)
        return self
            
class Metrics(dict):
    #if you want to add more metrics, add them here and it should have update and reset methods
    def __init__(self,device,num_classes):
        self.device = device
        self.metrics =  {"loss":AverageMeter(), "accuracy":AverageMeter(),"f1":MulticlassF1Score(task="multiclass", num_classes=num_classes,average="macro").to(device) } #"cm" :  SingleConfusionMatrix(task="multiclass",num_classes=num_classes).to(device) #MultiConfusionMatrix(task='binary',num_heads=40,num_classes=2).to(device) }#"pfpr": PrecesionFRR()
        self.best_accuracy = 0
        self.best_loss = np.inf
        for m in self.metrics.keys():
            setattr(self, m, self.metrics[m])
            

    def update(self,metric_input,num):
        for metric, metric_func in self.metrics.items():
            self.metrics[metric].update(metric_func(metric_input),num)
    def reset(self):
        for metric in self.metrics.keys():
            self.metrics[metric].reset()
    def __getitem__(self, item):
        return getattr(self, item)
    def items(self) -> dict_items:
        return self.metrics.items()
    
def test_model(model,dataloader,device,exp_name = "test",logger=None):
    
    #test phase
    model.eval()
    correct,masked,unmasked,masked_count,unmasked_count,total = 0,0,0,0,0,0
    totsl_lb = 0
    gradCam = GradCamWrapper(model,device)
    for batch_id,batch_data in tqdm(enumerate(dataloader)):
        images,labels = batch_data[0].to(device), batch_data[1].to(device)
        with torch.no_grad():
            outputs = model(images)
            if batch_id ==0 and logger is not None:
                logger.info(f"Outputs  {exp_name} : {outputs}")
            _, predicted = torch.max(outputs,dim= -1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            masked += (predicted[batch_data[2]] == labels[batch_data[2]]).sum()
            unmasked += (predicted[~batch_data[2]] == labels[~batch_data[2]]).sum()
            totsl_lb +=labels[batch_data[2]].sum()
            masked_count += batch_data[2].sum()
            unmasked_count += (~batch_data[2]).sum()

        if random.random() < 0.01:
            for id,filter in enumerate([batch_data[2],~batch_data[2]]):
                if filter.sum() == 0:
                    continue
                exp_fig,exp_ax = gradCam([batch_data[0][filter] for x in batch_data])
                exp_ax.set_title(f"(Mask,predicted,labels) : {[(i.item(),j.item(),l.item()) for i,j,l in zip(batch_data[2][filter],predicted[filter],labels[filter])]}",loc='center', wrap=True)
                exp_fig.savefig(f"logs/explainable/{id}_{exp_name}_{batch_id}.png")
                plt.close()

    print(f"Accuracy of the network on the {total} test images: {100 * correct / total} %%"  )
    print(f"Accuracy of the network on the {masked_count} test images: {100 * masked / masked_count} %%"  )
    print(f"Accuracy of the network on the {unmasked_count} test images: {100 * unmasked / unmasked_count} %%"  )
    print(totsl_lb)
    return 

class TrainerMetrics(dict):
    def __init__(self,device,num_classes,test=True):
        self.splits = {"train": Metrics(device,num_classes),"val":Metrics(device,num_classes)}
        if test: self.splits["test"] = Metrics(device,num_classes)
        for m in self.splits.keys():
            setattr(self, m, self.splits[m])
    def __getitem__(self, item):
        return getattr(self, item)
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    def items(self) -> dict_items:
        return self.splits.items()
    
def _call_if_vanila_verbose(func):
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper

def _call_if_verbose(func):
    def wrapper(self, *args, **kwargs):
        if self.verbose and (self.epoch) % self.log_freq == 0:
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper
    


    
class Trainer:
    def __init__(self,model: nn.Module, dataloaders: TrainerDataLoaders,trainer_args: TrainerSettings):


        self.name = trainer_args.name
        self.device = trainer_args.device
        self.num_classes = trainer_args.num_classes
        self.trainer_args = trainer_args
        

        self.model = model.to(self.device)
        self.gradCam = GradCamWrapper(self.model,self.device)
        
        
        self.dataloaders = dataloaders
        self.train_loader = dataloaders.train
        self.val_loader = dataloaders.val
        self.test_loader = dataloaders.test


        self.optimizer = operator.attrgetter(trainer_args.optimizer.type)(optim)(
                                    self.model.parameters(), **{k: v for k, v in trainer_args.optimizer.__dict__.items() if k!="type"})
        self.scheduler = operator.attrgetter( trainer_args.scheduler.type)(
                            tr)(self.optimizer, **{k: v for k, v in  trainer_args.scheduler.__dict__.items() if k!="type"})

        self.loss_fn = trainer_args.loss_fn   
        self.verbose = trainer_args.verbose

        self.init_trainer_tracker()
        self.model_dir = trainer_args.model_dir
        self.logger = logging.getLogger()
        self.log_path = trainer_args.log_path
        self.writer = SummaryWriter(self.log_path + f"/runs/{self.name}")
        self.log_freq = trainer_args.log_freq

    def train(self, epochs: int):
        
        progress_bar = tqdm(range(self.epoch+1, epochs+self.epoch+1))#, disable=not self.verbose
        start_time = time.time()
        for epoch in progress_bar:

            self.epoch = epoch
            epoch_time = time.time()
            

            train_epoch_time = time.time()
            self.train_epoch()
            self.logger.info(f"Train epoch time taken : {time.time()-train_epoch_time}")
            test_epoch_time = time.time()
            self.test_epoch()
            self.logger.info(f"Test epoch time taken : {time.time()-test_epoch_time}")
            logging_time = time.time()
            self.epoch_logging(epoch,progress_bar)
            if self.metrics.val.best_accuracy <= self.metrics.val.accuracy.avg:
                self.metrics.val.best_accuracy = self.metrics.val.accuracy.avg
                self.best_model = self.model.state_dict()
                self.save(self.model_dir+f"/model_{self.name}.pt")    

            #self.model.load_state_dict(self.best_model)
            self.logger.info(f"Best model accuracy: {self.metrics.val.best_accuracy}")
            self.logger.info(f"Best model is saved @ : {self.model_dir}/model_{self.name}.pt")
            self.logger.info(f"Epoch time taken : {time.time()-epoch_time}")
            self.logger.info(f"Logging time taken : {time.time()-logging_time}")


        self.logger.info(f"Total time taken : {time.time()-start_time}")
        return self.model
    

    def train_epoch(self):
        self.model.train()
        self.metrics.train.reset() 
        
        for batch_id, batch_data in enumerate(self.train_loader):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            outputs = self.model(inputs) #targets 
           
            loss = self.loss_fn(outputs, targets) 
            self.optimizer.zero_grad()      
            loss.backward()
            self.optimizer.step()

            #### LOGGING ######
            self.batch_logging(split = "train",batch_id = batch_id,batch_data = batch_data,outputs=outputs,loss = loss)
            ###################
        self.scheduler.step()

    
    @_call_if_vanila_verbose
    def test_epoch(self):
        self.metrics.val.reset() 
        self.test(self.val_loader, split="val")

        if self.test_loader is not None: 
            self.metrics.test.reset() 
            self.test(self.test_loader,split="test")

    
    def test(self, loader: DataLoader,split="test"):
        self.model.eval()
        for batch_id,batch_data in enumerate(loader):
            with torch.no_grad():
                inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            self.batch_logging(split,batch_id = batch_id,batch_data = batch_data,outputs=outputs,loss = loss)
            
    ############### Load and Save ############################
    def load(self, path):

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.history = checkpoint['history']
        return self.model

        
    def save(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.best_model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'history': self.history
        }, path)
    
    def load_from_checkpoint(self, path):
        trainer = self.__class__(copy.deepcopy(self.model), self.dataloaders, self.trainer_args)
        trainer.load(path)
        return trainer
    
    def reset(self):
        def  init_weights(mod):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
        #reset model weights
        self.model.apply(init_weights)
        self.optimizer.state = defaultdict(dict)
        self.scheduler.state = defaultdict(dict) 
        self.init_trainer_tracker()

    ############################# START Logging ########################################
    
    @_call_if_verbose
    def batch_logging(self,split,batch_id,batch_data,outputs,loss):
        self.log_metrics(split,batch_id,batch_data,outputs,loss)
        #self.save(self.model_dir+f"/model_{self.name}.pt")
        #self.log_gradients(split)
        #self.random_data_snap(split,batch_id,batch_data,outputs,loss)
        #self.random_grad_cam(split,batch_id,batch_data,outputs,loss)
    
    @_call_if_verbose
    def epoch_logging(self,epoch,progress_bar):
        metric_dict =  {"lr": self.scheduler.get_last_lr()[0]}
        for split,metrics in self.metrics.items():
            for metric_name,metric in metrics.items():
                if  metric_name =="cm": 
                    metrics.cm.plot(filename=os.path.join(self.log_path,f"{split}_cm_{self.epoch}.png"))
                    continue
                elif metric_name in[ "accuracy","loss"]:
                    metric_dict[f"{split}_{metric_name}"] = metric.avg
                    metric_dict[f"{split}_count"] = metric.count
                else:
                    metric_dict[f"{split}_{metric_name}"] = metric.compute().item()

                if split=="val" and metric_name == "loss": 
                    self.best_loss = min(self.best_loss,metric.avg)
                if split=="val" and metric_name == "accuracy" and metric.avg >= self.best_accuracy: 
                    self.best_accuracy = metric.avg
                    self.best_model = self.model.state_dict()
                
        self.logger.info(f"Epoch {epoch} logging : {metric_dict}")
        self.log(epoch,metric_dict,progress_bar)

    
    
    @_call_if_verbose
    def log(self, epoch, logs,progress_bar=None):                    
        def group(x): return x.split("_")[-1]
        for key, value in logs.items():
            self.history[key].append(value)
            self.writer.add_scalar(f"{group(key)}/{key}", value, epoch)
        if progress_bar is not None: progress_bar.set_postfix(**logs)
        
    
    @_call_if_verbose
    def  random_data_snap(self,split,batch_id,batch_data,outputs,loss):
        if random.random() < 0.01:
            plot_image_grid(batch_data[0][:16],labels=batch_data[1][:16], filename=self.log_path+f"/{split}_{self.epoch}_{batch_id}.png",title=f"Epoch {self.epoch} {split} Images")
    
    @_call_if_verbose
    def random_grad_cam(self,split,batch_id,batch_data,outputs,loss):
        if random.random() < 0.01:
            exp_fig,exp_ax = self.gradCam(batch_data)
            exp_ax.set_title(f"(Mask,predicted,labels) : {[(i.item(),j.item(),l.item()) for i,j,l in zip(batch_data[2],torch.argmax(outputs,dim=1),batch_data[1])]}",loc='center', wrap=True)
            exp_fig.savefig(self.log_path+f"/gradCam_{split}_{self.epoch}_{batch_id}.png")
            plt.close()
    
    @torch.no_grad()
    def log_metrics(self,split,batch_id,batch_data,outputs,loss):
        inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
        softmax = nn.Softmax(dim=1)
        for metric_name,metric_fn in self.metrics[split].items():
            if metric_name == "loss":
                metric_fn.update(loss.item(), inputs.size(0))
            elif metric_name == "accuracy":
                metric_fn.update(self.accuracy(outputs, targets), inputs.size(0))
            elif metric_name == "cm":
                metric_fn.update(torch.argmax(outputs, dim=-1), targets)
            else:

                metric_fn.update(F.softmax(outputs,dim=-1),targets)

    @_call_if_verbose
    def log_gradients(self,split):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"{split}.{name}.grad", param.grad, self.epoch)


    ############################# END Logging ########################################

    def accuracy(self, outputs, targets):
        preds = torch.argmax(outputs, dim=-1)
        return  (preds == targets).float().mean().item()

    def cross_validation_score(self, epochs=100):
        self.train(epochs)
        self.test_epoch()
        return self.metrics["val"]["accuracy"].avg
    

    def init_trainer_tracker(self):
        self.history = defaultdict(list)
        self.epoch = -1
        self.best_accuracy = 0
        self.best_loss = np.inf
        self.best_model = self.model.state_dict()
        self.metrics = TrainerMetrics(self.device,self.num_classes,test=self.test_loader is not None)



class NNTrainer(Trainer):
    def __init__(self, model: nn.Module, dataloaders: TrainerDataLoaders, trainer_args: TrainerSettings):
        
        super().__init__(model, dataloaders, trainer_args)
        self.hook_features = None
        self.og_model = copy.deepcopy(self.model)
        self.hook_handle = self.setup_hook()
        self.nn_optim = copy.deepcopy(self.optimizer)
        
    def setup_hook(self):
        def forward_hook_fn(module, inputs, outputs):
            self.hook_features = outputs
        handle = self.og_model.fc.register_forward_hook(forward_hook_fn)
        return handle
    
    def nn_epoch(self):

        self.model.train()       
        retain_iter = iter(self.dataloaders.retain)
        for batch_id, batch_data in enumerate(self.dataloaders.forget):
            inputs_forget, targets_forget  = batch_data[0].to(self.device), batch_data[1].to(self.device)
            try:
                batch_data_retain = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.dataloaders.forget)
                batch_data_retain = next(retain_iter)
            
            inputs_retrain, targets_retrain = batch_data_retain[0].to(self.device), batch_data_retain[1].to(self.device)
            inputs = torch.cat([inputs_retrain,inputs_forget],dim=0)
            targets = torch.cat([targets_retrain,targets_forget],dim=0)
            retain_mask_og = torch.cat([torch.ones_like(targets_retrain),torch.zeros_like(targets_forget)],dim=0)

            outputs = self.model(inputs) 

            targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
            retain_mask = retain_mask_og.reshape(-1,1).float()

            with torch.no_grad():
                _ = self.og_model(inputs) # only for hook_features
                hk = self.hook_features
                D = -torch.norm(hk[:, None] - hk, dim=-1)
                D[:,retain_mask.flatten()==0] = -100000
                ans =  F.softmax( D - (1 - retain_mask @ retain_mask.T + torch.eye(hk.shape[0],device=self.device))*100000 ,dim=1) @ targets_onehot
                targets_onehot =  (retain_mask * targets_onehot + (1-retain_mask) * ans)
            
            loss = self.loss_fn(outputs, targets_onehot) 

            self.nn_optim.zero_grad()      
            loss.backward()
            self.nn_optim.step()


    def train(self, epochs: int):
        
        progress_bar = tqdm(range(self.epoch+1, epochs+self.epoch-1))#, disable=not self.verbose
        start_time = time.time()

        for i in range(20):
            self.nn_epoch()
        
        self.dataloaders.train = self.dataloaders.retain
        self.train_loader = self.dataloaders.retain

        for epoch in progress_bar:

            self.epoch = epoch
            epoch_time = time.time()

            train_epoch_time = time.time()
            self.train_epoch()
            self.logger.info(f"Train epoch time taken : {time.time()-train_epoch_time}")
            test_epoch_time = time.time()
            self.test_epoch()
            self.logger.info(f"Test epoch time taken : {time.time()-test_epoch_time}")
            logging_time = time.time()
            self.epoch_logging(epoch,progress_bar)
            if self.metrics.val.best_accuracy <= self.metrics.val.accuracy.avg:
                self.metrics.val.best_accuracy = self.metrics.val.accuracy.avg
                self.best_model = self.model.state_dict()
                self.save(self.model_dir+f"/model_{self.name}.pt")    

            #self.model.load_state_dict(self.best_model)
            self.logger.info(f"Best model accuracy: {self.metrics.val.best_accuracy}")
            self.logger.info(f"Best model is saved @ : {self.model_dir}/model_{self.name}.pt")
            self.logger.info(f"Epoch time taken : {time.time()-epoch_time}")
            self.logger.info(f"Logging time taken : {time.time()-logging_time}")


        self.logger.info(f"Total time taken : {time.time()-start_time}")
        return self.model


class AdverserialTrainer(Trainer):
    def __init__(self, model: nn.Module, dataloaders: TrainerDataLoaders, trainer_args: TrainerSettings):
        super().__init__(model, dataloaders, trainer_args)

        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        self.adv_loss = nn.KLDivLoss(reduction="batchmean")
        
        #from the paper
        self.alpha = 0.1 
        self.gamma = 3
        self.max_epochs = 1

    def train_epoch(self):

        self.model.train()      
        for _ in range(self.max_epochs):
            self.max_epoch(self.dataloaders.forget)
        self.min_epoch(self.dataloaders.retain)
        self.scheduler.step()
            
    def min_epoch(self,dataloader):

        for batch_id, batch_data in enumerate(dataloader):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            
            outputs = self.model(inputs)
            loss = self.gamma * self.loss_fn(outputs,targets) + self.alpha*self.adv_loss(F.softmax(outputs,dim=1), F.softmax(self.teacher_model(inputs),dim=1))
            
            #### LOGGING ######
            self.batch_logging("train",batch_id = batch_id,batch_data = batch_data,outputs=outputs,loss = loss)
            ###################
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def max_epoch(self,dataloader):

        for batch_id, batch_data in enumerate(dataloader):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            outputs = self.model(inputs)
            loss = -1*self.adv_loss(F.softmax(outputs,dim=1), F.softmax(self.teacher_model(inputs),dim=1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




class RLFTrainer(NNTrainer):
    def __init__(self, model: nn.Module, dataloaders: TrainerDataLoaders, trainer_args: TrainerSettings):
        super().__init__(model, dataloaders, trainer_args)
        self.model = copy.deepcopy(self.model)


    def nn_epoch(self):
        self.model.train()
        self.dataloaders.train = self.dataloaders.retain
        for batch_id, batch_data in enumerate(self.dataloaders.forget):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            targets = targets[torch.randperm(targets.size()[0])]
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs,targets)
            self.nn_optim.zero_grad()
            loss.backward()
            self.nn_optim.step()
  

def discretize(x):
    return torch.round(x * 255) / 255

class BoundaryUnlearning(NNTrainer):
    def __init__(self, model: nn.Module, dataloaders: TrainerDataLoaders, trainer_args: TrainerSettings):
        super().__init__(model, dataloaders, trainer_args)
        self.eps = 0.1
        self.nn_optimizer = copy.deepcopy(self.optimizer)
    
    def FGSM_perturb(self,x, y, model=None, bound=None, criterion=None):
        device = model.parameters().__next__().device
        model.zero_grad()
        x_adv = x.detach().clone().requires_grad_(True).to(device)

        pred = model(x_adv)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = x_adv + grad_sign * bound
        x_adv = discretize(torch.clamp(x_adv, 0.0, 1.0))

        return x_adv.detach()

    def nn_epoch(self):
        self.dataloaders.train = self.dataloaders.retain
        #gradients to inputs
        self.model.train()
        for batch_id, batch_data in enumerate(self.dataloaders.forget):
            inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)

            mod_inputs = self.FGSM_perturb(inputs,targets,self.model,self.eps,self.loss_fn)
            new_targets = torch.argmax(self.model(mod_inputs).detach(), dim=1)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs,new_targets)

            self.nn_optimizer.zero_grad()
            loss.backward()
            self.nn_optimizer.step()
        




