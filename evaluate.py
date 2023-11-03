from torch import nn, optim
import torch
from torchvision.models import resnet18,ResNet18_Weights
from torch.utils.data.sampler import WeightedRandomSampler
from celeba_dataset import get_dataset
from tqdm import tqdm
import math

from torchmetrics import Accuracy
import os
import  numpy as np
from attacks import ThresholdAttack


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ATTACKS =[ThresholdAttack(i/10.0) for i in range(1,10)]

def f_score(retain_loader,forget_loader,valid_loader,model,original_checkpoints_path,unlearned_checkpoints_path,device):
    #load all the models and get the 512 x 8 weight matrix
    
    original_logits = prepare_logits(forget_loader,model,original_checkpoints_path,device)
    unlearn_logits = prepare_logits(forget_loader,model,unlearned_checkpoints_path,device)
    #save numpy of logits
    np.save('logits/original_logits.npy',original_logits.cpu().numpy())
    np.save('logits/unlearn_logits.npy',unlearn_logits.cpu().numpy())
    utility = calculate_utility(retain_loader,valid_loader,model,original_checkpoints_path,unlearned_checkpoints_path,device)
    
    return utility

def H_score(original_logits,unlearn_logits):
    #512x8
    
    delta = 0.1
    H = []
    for og,ul in zip(original_logits,unlearn_logits):
        per_attack_e = []
        for attack in ATTACKS:
            fpr,fnr = attack(og,ul) 
            if fpr ==0 and fnr ==0:
                per_attack_e.append(np.Inf)
            elif fpr ==0 or fnr ==0:
                pass
            else:
                per_attack_e1 = np.log(1-delta - fpr) - np.log(fnr)
                per_attack_e2 = np.log(1-delta - fnr) - np.log(fpr)
                per_attack_e.append(max(per_attack_e1,per_attack_e2))
        per_attack_e = np.array(per_attack_e)
        per_attack_e = per_attack_e[~np.isnan(per_attack_e)]
        e = np.nanmax(per_attack_e) if len(per_attack_e) > 0 else 0
        h = 2.0 / 2**(math.floor(e*2))
        H.append(h)
    return np.mean(H)

    


def prepare_logits(data_loader,model,check_point_path,device):
    model.eval()
    logits = []
    for model_ckpt in os.listdir(check_point_path)[:10]:
        model.load_state_dict(torch.load(os.path.join(check_point_path,model_ckpt)))
        logits.append(get_logits(data_loader,model,device).unsqueeze(1))
    return torch.cat(logits,dim=1)

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
    acc = Accuracy(task="multiclass",num_classes=8).to(device)
    for sample in data_loader:
        inputs,targets = sample["image"].to(device), sample["age_group"].to(device)       
        outputs = model(inputs)
        acc.update(outputs,targets)
    return acc.compute()



def average_accuracy(data_loader,model,model_checkpoint_path,device):
    instance_accuracy= []
    for model_ckpt in tqdm(os.listdir(model_checkpoint_path)[:10]):
        model.load_state_dict(torch.load(os.path.join(model_checkpoint_path,model_ckpt)))
        acc = compute_accuracy(data_loader,model,device)
        instance_accuracy.append(acc)
    return sum(instance_accuracy)/len(instance_accuracy)

def calculate_utility(retain_loader, valid_loader, model,model_checkpoint_path,unlearned_checkpoints_path,device):
    RA_R = average_accuracy(retain_loader,model,model_checkpoint_path,device)
    TA_R = average_accuracy(valid_loader,model,model_checkpoint_path,device)
    RA_U = average_accuracy(retain_loader,model,unlearned_checkpoints_path,device)
    TA_U = average_accuracy(valid_loader,model,unlearned_checkpoints_path,device)
    return RA_U*TA_U / (RA_R*TA_R)


if __name__ == "__main__":
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512,8)
    model.to(DEVICE)
    train_loader,retain_loader, forget_loader, validation_loader = get_dataset(32)
    print(f_score(retain_loader,forget_loader,validation_loader,model,'neurips-2023-machine-unlearning','tmp',DEVICE))
    """
    original_logits = np.load('logits/original_logits.npy')
    unlearn_logits = np.load('logits/unlearn_logits.npy')
    print(H_score(original_logits=original_logits,unlearn_logits=unlearn_logits))




    



    