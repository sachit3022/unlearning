from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from typing import List,Optional



import torch
torch.cuda.empty_cache()
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,Dataset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18



from config import Config
from network import ResNet,AverageMeter
from score import compute_mia_score
from plots import plot_losses
from torch.utils.tensorboard import SummaryWriter


DEVICE = Config.DEVICE.value



def compute_losses_and_logits(model,criterion, loader):
    """Auxiliary function to compute per-sample losses"""
        
    all_losses = []
    all_logits = []
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = model(inputs)
        losses = criterion(logits, targets).numpy(force=True)
       
        for loss,logit in zip(losses,logits):
            all_losses.append(loss)
            all_logits.append(logit.numpy(force=True))

    return np.array(all_losses),np.array(all_logits)

def compute_unlearning_metrics(full_model,forget_model,forget_loader,test_loader):
    """Compute unlearning metrics. Current focus in MIA metric."""

    loss_criterion_per_sample = nn.CrossEntropyLoss(reduction="none")
    
    test_losses, test_logits = compute_losses_and_logits(full_model, loss_criterion_per_sample, test_loader)
    before_forget_losses, before_forget_logits = compute_losses_and_logits(full_model, loss_criterion_per_sample, forget_loader)
    mia_before = compute_mia_score(test=test_logits,forget=before_forget_logits)
    #save the losses and logits
    np.savez(os.path.join(Config.RESULTS_PATH.value,"before_losses.npz"),test_losses=test_losses,forget_logits=before_forget_losses)
    np.savez(os.path.join(Config.RESULTS_PATH.value,"before_logits.npz"),test_logits=test_logits,forget_logits=before_forget_logits)

    test_losses, test_logits = compute_losses_and_logits(forget_model, loss_criterion_per_sample, test_loader)
    after_forget_losses, after_forget_logits = compute_losses_and_logits(forget_model, loss_criterion_per_sample, forget_loader)    
    mia_after = compute_mia_score(test=test_logits,forget=after_forget_logits)

    #save the losses and logits
    np.savez(os.path.join(Config.RESULTS_PATH.value,"after_losses.npz"),test_losses=test_losses,forget_logits=after_forget_losses)
    np.savez(os.path.join(Config.RESULTS_PATH.value,"after_logits.npz"),test_logits=test_logits,forget_logits=after_forget_logits)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_losses(ax[0], test_losses,before_forget_losses)
    plot_losses(ax[1],test_losses,after_forget_losses)
    fig.savefig(Config.RESULTS_PATH.value + "/loss_patterns.png")
    
    return mia_before,mia_after


def count_parameters(model): 
    #copied from pytorch disccussion forum
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self,name:str,model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,test_loader:Optional[DataLoader],optimizer:optim.Optimizer,scheduler , loss_fn:nn.Module,device:str):
        
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.name = name

        self.loss_fn = loss_fn
        self.device = device


        self.train_loss,self.train_accuracy = AverageMeter(),AverageMeter()
        self.test_loss,self.test_accuracy = AverageMeter(),AverageMeter()
        self.val_loss,self.val_accuracy = AverageMeter(),AverageMeter()

        
        self.epoch = -1
        self.best_accuracy = 0
        self.best_loss = np.inf
        self.history = defaultdict(list)
        self.writer = SummaryWriter(Config.LOG_PATH.value  + f"/runs/{self.name}")

    def train(self,epochs:int):
        for epoch in tqdm(range(self.epoch+1,epochs+self.epoch+1)):
            self.epoch = epoch
            self.train_epoch()
            self.test_epoch()
            self.log(epoch,{"train_loss":self.train_loss.avg,"test_loss":self.test_loss.avg,"train_accuracy":self.train_accuracy.avg,
                            "test_accuracy":self.test_accuracy.avg,"lr":self.scheduler.get_last_lr()[0],
                            "val_loss":self.val_loss.avg,"val_accuracy":self.val_accuracy.avg
                            })
            if self.test_accuracy.avg >= self.best_accuracy:
                self.best_accuracy = self.test_accuracy.avg
                self.best_loss = self.test_loss.avg
                self.best_model = self.model.state_dict()


        self.model.load_state_dict(self.best_model)
        self.save(os.path.join(Config.MODEL_DIR.value,f"weigths_{self.name}.pt"))
        return self.model
    
    def log(self,epoch,logs):
        for key,value in logs.items():
            self.history[key].append(value)
            self.writer.add_scalar(f"{key}_{self.name}",value,epoch)


    def train_epoch(self):
        self.model.train()
        self.train_loss.reset()
        self.train_accuracy.reset()
        for inputs,targets in self.train_loader:
            inputs,targets = inputs.to(self.device),targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs,targets)
            self.train_loss.update(loss.item(),inputs.size(0))
            self.train_accuracy.update(self.accuracy(outputs,targets),inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def test_epoch(self):
        self.test(self.val_loader,self.val_loss,self.val_accuracy)
        self.test(self.test_loader,self.test_loss,self.test_accuracy)
        self.debug(self.test_loader,debug_id=self.epoch)
            
    def test(self,loader:DataLoader,test_loss:AverageMeter,test_accuracy:AverageMeter):
        self.model.eval()
        test_loss.reset()
        test_accuracy.reset()
        with torch.no_grad():
            for inputs,targets in loader:
                inputs,targets = inputs.to(self.device),targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs,targets)
                test_loss.update(loss.item(),inputs.size(0))
                test_accuracy.update(self.accuracy(outputs,targets),inputs.size(0))

    def save(self,path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.best_model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'history': self.history
            }, path)
    
    def load(self,path):
        
        checkpoint = torch.load(path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.history = checkpoint['history']
        return self.model
    
    def debug(self,test_loader,debug_id=0):
        self.model.eval()
        debug_data = defaultdict(list)

        for batch_idx, (inputs,targets) in  enumerate(test_loader):
            inputs,targets = inputs.to(self.device),targets.to(self.device)
            probs = self.model.predict_pobs (inputs)

            debug_predictions = torch.arange(batch_idx*inputs.size(0),(batch_idx+1)*inputs.size(0))
            #debug_predictions =  (torch.argmax(probs,dim=1) != targets).nonzero().squeeze().detach() +batch_idx # only if we want to log incorrect predictions
            debug_data["incorrect_predictions"].append(debug_predictions)
            debug_data["targets"].append(targets[debug_predictions].detach())
            debug_data["probs"].append(probs[debug_predictions].detach())
        
        log_path = os.path.join(Config.LOG_PATH.value,self.name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({ name : torch.cat(each_tensors,dim=0) for name,each_tensors in debug_data.items()}, os.path.join(log_path, f"debug_predictions_{debug_id}.pt"))
        
        return 

    def accuracy(self,outputs,targets):
        _,preds = torch.max(outputs,dim=1)
        return torch.sum(preds == targets).item() / len(preds)
    
def main():

    #load the dataset
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH.value, train=True, download=True, transform=normalize
    )
    train_loader = DataLoader(train_set, batch_size=8192, shuffle=True, num_workers=8, pin_memory=True)  

    forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9])

    forget_loader = DataLoader(forget_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
    retain_loader = DataLoader(retain_set, batch_size=8192, shuffle=True, num_workers=8, pin_memory=True)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH.value, train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])

    test_loader = DataLoader(test_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
        
    models = ["/research/hal-gaudisac/unlearning/models/weigths_full_ResNet_3x16-2x32-2x64.pt","/research/hal-gaudisac/unlearning/models/weigths_forget_ResNet_3x16-2x32-2x64.pt"]
    layers = [(3,16),(2,32),(2,64)]


    for mode,loader in [("full",train_loader),("forget",retain_loader)]:
        
        #initialize the model
        layers = [(3,16),(2,32),(2,64)] 
        epochs = 100
        net = ResNet(layers)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        trainer = Trainer(f"{mode}_"+net.name,net,loader,val_loader,test_loader,optimizer,scheduler,criterion,DEVICE)
        model = trainer.train(epochs)
        model.eval()
        model_path = os.path.join(Config.MODEL_DIR.value,f"weigths_{mode}_{model.name}.pt")
        models.append(model_path)
        trainer.save(model_path)

        
    #load the models
    for path in models:
        checkpoint = torch.load(path)
        
        model = ResNet(layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        if "full" in path:
            full_model = model
        else:
            forget_model = model
    
    #unlearning metrics
    mia_before, mia_after = compute_unlearning_metrics(full_model,forget_model,forget_loader,test_loader)
    print(f"The MIA attack has an accuracy of {mia_before.mean():.3f} before unlearning and {mia_after.mean():.3f} after unlearning on forgotten vs unseen images")
    return 


def test_resnet18_on_cifar(checkpoint =None,train=False):
    train_transforms = transforms.Compose(               
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train_set = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH.value, train=True, download=False, transform=train_transforms
    )
    train_loader = DataLoader(train_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)  
    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH.value, train=False, download=False, transform=test_transforms
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])

    test_loader = DataLoader(test_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
        
    #initialize the model

    layers = [(3,16),(2,32),(2,64)] 
    epochs = 100
    net = ResNet(layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"Model has {count_parameters(net)} parameters")

    trainer = Trainer(net.name,net,train_loader,val_loader,test_loader,optimizer,scheduler,criterion,DEVICE)
    
    if checkpoint is not None:
        trainer.load(checkpoint)
    if train:
        trainer.train(epochs)

    forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9])
    forget_loader = DataLoader(forget_set, batch_size=8192, shuffle=False, num_workers=8, pin_memory=True)
    retain_loader = DataLoader(retain_set, batch_size=8192, shuffle=True, num_workers=8, pin_memory=True)
    


    #how forgetting happens in the model when finetuining.
    epochs = 100
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4) # the fine tuning loss should be very small.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    finetune_trainer = Trainer("finetune_"+net.name,trainer.model,retain_loader,val_loader,forget_loader,optimizer,scheduler,criterion,DEVICE)
    finetune_trainer.train(100)
    """
    for i in range(50):
        test_acc = finetune_trainer.predict(test_loader ,debug=False) 
        forget_acc = finetune_trainer.predict(forget_loader,debug=True,debug_id=i+1)
        print(f"{i} -> forget acc:  is {forget_acc:.3f} and test acc is {test_acc:.3f}")
        finetune_trainer.train(1)  
    """
    return



if __name__ == "__main__":
    
    """"
    #pretrained one
    # load model with pre-trained weights
    weights_pretrained = torch.load("weights_resnet18_cifar10.pth", map_location=DEVICE)
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval();
    """
    #main()
    test_resnet18_on_cifar(os.path.join(Config.MODEL_DIR.value,"weigths_ResNet_3x16-2x32-2x64.pt"),train=False)
    
