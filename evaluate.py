from torch_metrics import Accuracy
from torch import nn, optim
import torch
from torchmetrics import Accuracy
import os
import  numpy as np
ATTACKS = ["FGSM","PGD","CW"]


def f_score(retain_loader,forget_loader,valid_loader,model,original_checkpoints_path,unlearned_checkpoints_path,device):
    #load all the models and get the 512 x 8 weight matrix
    #Nx512x8 
    original_logits = prepare_logits(forget_loader,model,original_checkpoints_path,device)
    unlearn_logits = prepare_logits(forget_loader,model,unlearned_checkpoints_path,device)

        
    
    utility = calculate_utility(retain_loader,valid_loader,model,original_checkpoints_path,unlearned_checkpoints_path,device)
    return utility

def F_score_for_a_sample(original_logits,unlearn_logits):
    #512x8
    per_attack_e = []
    delta = 0.1
    for attack in ATTACKS:
        fpr,fnr= compute_fpr(original_logits,unlearn_logits,attack)
        if fpr ==0 and fnr ==0:
            per_attack_e.append(np.Inf)
        elif fpr ==0 or fnr ==0:
            pass
        else:
            per_attack_e1 = np.log(1-delta - fpr) - np.log(fnr)
            per_attack_e2 = np.log(1-delta - fnr) - np.log(fpr)
            per_attack_e.append(max(per_attack_e1,per_attack_e2))
    

def compute_fpr(original_logits,unlearn_logits,attack):
    attack.fit(original_logits,unlearn_logits)
    fpr = attack.false_positive_rate()
    fnr = attack.false_negative_rate()
    return fpr,fnr

def prepare_logits(data_loader,model,check_point_path,device):
    model.eval()
    logits = []
    for model_ckpt in os.listdir(check_point_path):
        model.load_state_dict(torch.load(model_ckpt))
        logits.append(get_logits(data_loader,model,device))
    return torch.cat(logits,dim=0)

@torch.no_grad()
def get_logits(data_loader,model,device):
    model.eval()
    logits = []
    for sample in data_loader:
        inputs = sample["image"]
        targets = sample["age_group"]
        inputs, targets = inputs.to(device), targets.to(device)               
        outputs = model(inputs)
        logits.append(outputs)
    return torch.cat(logits,dim=0)


@torch.no_grad()
def compute_accuracy(data_loader,model,device):
    model.eval()
    #load all the models and get the 512 x 8 weight matrix
    acc = Accuracy(task="multiclass",num_classes=8).to(device)
    for sample in data_loader:
        inputs = sample["image"]
        targets = sample["age_group"]
        inputs, targets = inputs.to(device), targets.to(device)               
        outputs = model(inputs)
    return acc.update(outputs,targets)


def average_accuracy(data_loader,model,model_checkpoint_path,device):
    instance_accuracy= []
    for model_ckpt in os.listdir(model_checkpoint_path):
        model.load_state_dict(torch.load(model_ckpt))
        acc = compute_accuracy(data_loader,model,device)
        instance_accuracy.append(acc)
    return sum(instance_accuracy)/len(instance_accuracy)

def calculate_utility(retain_loader, valid_loader, model,model_checkpoint_path,unlearned_checkpoints_path,device):
    RA_R = average_accuracy(retain_loader,model,model_checkpoint_path,device)
    TA_R = average_accuracy(valid_loader,model,model_checkpoint_path,device)
    RA_U = average_accuracy(retain_loader,model,unlearned_checkpoints_path,device)
    TA_U = average_accuracy(valid_loader,model,unlearned_checkpoints_path,device)
    return RA_U*TA_U / (RA_R*TA_R)




    



    