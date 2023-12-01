from torch import nn, optim
import torch
from torchvision.models import resnet18,ResNet18_Weights
from celeba_dataset import get_dataset
from tqdm import tqdm
import math

import os
import  numpy as np
from attacks import ThresholdAttack,GaussianAttack

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load all the models and get the 512 x 8 weight matrix
def make_logits(retain_loader,forget_loader,val_loader,model,original_checkpoints_path,unlearned_checkpoints_path,device,path = "logits"):
    os.makedirs(path,exist_ok=True)
    for name,loader in tqdm([("retain",retain_loader),("forget",forget_loader),("val",val_loader)]):
        targets = prepare_targets(loader)
        np.save(f"{path}/{name}_targets.npy",targets.cpu().numpy())
    for name,loader,chkpt_path in tqdm([("retrain_forget_logits",forget_loader,original_checkpoints_path),("unlearn_forget_logits",forget_loader,unlearned_checkpoints_path), ("retrain_retain_logits",retain_loader,original_checkpoints_path),("unlearn_retain_logits",retain_loader,unlearned_checkpoints_path),("retrain_val_logits",val_loader,original_checkpoints_path),("unlearn_val_logits",val_loader,unlearned_checkpoints_path)]):
        logits = prepare_logits(loader,model,chkpt_path,device)
        np.save(f"{path}/{name}.npy",logits.cpu().numpy())
    return 


def prepare_logits(data_loader,model,check_point_path,device):
    model.eval()
    logits = []
    for model_ckpt in os.listdir(check_point_path):
        model_checkpoint = torch.load(f"{check_point_path}/{model_ckpt}", map_location=device)
        model.load_state_dict(model_checkpoint["model_state_dict"])
        logits.append(get_logits(data_loader,model,device).unsqueeze(1))
    return torch.cat(logits,dim=1)

def prepare_targets(data_loader):
    targets = []
    for sample in data_loader:
        targets.append(sample[1])
    return torch.cat(targets,dim=0)

@torch.no_grad()
def get_logits(data_loader,model,device):
    model.eval()
    logits = []
    for sample in data_loader:
        inputs = sample[0]
        targets = sample[1]
        inputs, targets = inputs.to(device), targets.to(device)               
        outputs = model(inputs)
        logits.append(outputs)

    return torch.cat(logits,dim=0)

ATTACKS =[ThresholdAttack(i/10.0) for i in range(1,10)]
ATTACKS.append(GaussianAttack())

class UnlearningScore:
    def __init__(self,logits_path ="logits") -> None:
        self.logits_path = logits_path
    def f_score(self):
        retrain_forget_logits =  np.load(f"{self.logits_path}/retrain_forget_logits.npy")
        unlearn_forget_logits = np.load(f"{self.logits_path}/unlearn_forget_logits.npy")
        retrain_retain_logits =  np.load(f"{self.logits_path}/retrain_retain_logits.npy")
        unlearn_retain_logits =  np.load(f"{self.logits_path}/unlearn_retain_logits.npy")
        retrain_val_logits =  np.load(f"{self.logits_path}/retrain_val_logits.npy")
        unlearn_val_logits = np.load(f"{self.logits_path}/unlearn_val_logits.npy")
        retain_targets =  np.load(f"{self.logits_path}/retain_targets.npy")
        val_targets = np.load(f"{self.logits_path}/val_targets.npy")
        self.utility = self.calculate_utility(retrain_retain_logits,retrain_val_logits,unlearn_retain_logits,unlearn_val_logits,retain_targets,val_targets)
        self.h_score = self.H_score(retrain_forget_logits,unlearn_forget_logits)
        return self.utility * self.h_score 

    def H_score(self,original_logits,unlearn_logits):
        #512x8
        delta = 0.1
        H = []
        for og,ul in zip(original_logits,unlearn_logits):
            per_attack_e = []
            all_samples_fpr,all_samples_fnr = [],[]
            for attack in ATTACKS:
                fpr,fnr = attack(og,ul) 
                all_samples_fpr.append(fpr)
                all_samples_fnr.append(fnr)
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
            e = np.nanmax(per_attack_e) if len(per_attack_e) > 0 else -np.Inf
            e = (1 / (1 + math.exp(-e)))*12 + 1
            h = 2.0 / 2**(e)
            H.append(h)
        return np.mean(H)

    def accuracy_across_models(self,logits,targets):
        logits = np.argmax(logits,axis=2)
        targets = np.expand_dims(targets,axis=1)
        return (logits == targets).mean()

    def calculate_utility(self,retrain_retain_logits,retrain_val_logits,unlearn_retain_logits,unlearn_val_logits,retain_targets,val_targets):
        RA_R = self.accuracy_across_models(retrain_retain_logits,retain_targets)
        TA_R = self.accuracy_across_models(retrain_val_logits,val_targets)
        RA_U = self.accuracy_across_models(unlearn_retain_logits,retain_targets)
        TA_U = self.accuracy_across_models(unlearn_val_logits,val_targets)
        return RA_U*TA_U / (RA_R*TA_R)


if __name__ == "__main__":
    

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512,4)
    model.to(DEVICE)
  
    train_loader,retain_loader, forget_loader, validation_loader ,test_loader= get_dataset(512)
    make_logits(retain_loader=retain_loader,forget_loader=forget_loader,val_loader=validation_loader,model=model,original_checkpoints_path='neurips-2023-machine-unlearning/retrain',unlearned_checkpoints_path='neurips-2023-machine-unlearning/scrubs',device=DEVICE)

    unl_score = UnlearningScore(logits_path="logits")    
    print(unl_score.f_score())
    print(unl_score.h_score,unl_score.utility)

    """
    0.017489625193614775
    0.017580117639039806 0.9948525688346888

    0.017122298220902513
    0.017209609738552 0.9949265835207234

    0.01765690397211273
    0.017612152175312486 1.002540961283708


    """



    