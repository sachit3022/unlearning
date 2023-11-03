import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18,ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import Accuracy
from celeba_dataset import get_dataset

from unl_algorithms import finetune_unlearning as unlearning


from celeba_dataset import UnlearnCelebADataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# It's really important to add an accelerator to your notebook, as otherwise the submission will fail.
# We recomment using the P100 GPU rather than T4 as it's faster and will increase the chances of passing the time cut-off threshold.
if DEVICE != torch.device("cuda:0"):
    raise RuntimeError('Make sure you have added an accelerator to your notebook; the submission will fail otherwise!')

# Helper functions for loading the hidden dataset.

def train(net, train_loader,val_loader,path):
    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()
    for ep in range(epochs):
        net.train()
        for sample in train_loader:
            inputs = sample["image"]
            targets = sample["age_group"]
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            acc = Accuracy(task="multiclass",num_classes=8).to(DEVICE)
            for sample in val_loader:
                inputs = sample["image"]
                targets = sample["age_group"]
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)               
                outputs = net(inputs)
                acc.update(outputs,targets)
            print(f"Epoch {ep} accuracy {acc.compute()}")
        torch.save(model.state_dict(), path)
    net.eval()

if __name__ == '__main__':

    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512,8)
    model.to(DEVICE)
    train_loader,retain_loader, forget_loader, validation_loader = get_dataset(32)

    train(model,train_loader,validation_loader,'neurips-2023-machine-unlearning/original_model.pth')
    torch.save(model.state_dict(), 'neurips-2023-machine-unlearning/original_model.pth')

    for i in range(512):
        model.load_state_dict(torch.load('neurips-2023-machine-unlearning/original_model.pth'))
        unlearning(model, retain_loader, forget_loader, validation_loader)
        state = model.state_dict()
        torch.save(state, f'tmp/unlearned_checkpoint_{i}.pth')
    """

    ##### RETRAIN CODE #####
    train_loader,retain_loader, forget_loader, validation_loader = get_dataset(32)
    for i in range(512):
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512,8)
        model.to(DEVICE)
        train(model,retain_loader,validation_loader,f"retrain/retrain_checkpoint_{i}.pth")
    ########################
    
    """
    #evaluation code
    if os.path.exists('neurips-2023-machine-unlearning/empty.txt'):
        # mock submission
        subprocess.run('touch submission.zip', shell=True)
    else:
        
        # Note: it's really important to create the unlearned checkpoints outside of the working directory 
        # as otherwise this notebook may fail due to running out of disk space.
        # The below code saves them in /kaggle/tmp to avoid that issue.
        
        os.makedirs('tmp', exist_ok=True)
        os.makedirs('neurips-2023-machine-unlearning', exist_ok=True)
        retain_loader, forget_loader, validation_loader = get_dataset(64)
        net = resnet18(weights=None, num_classes=10)
        net.to(DEVICE)
        for i in range(512):
            net.load_state_dict(torch.load('neurips-2023-machine-unlearning/original_model.pth'))
            unlearning(net, retain_loader, forget_loader, validation_loader)
            state = net.state_dict()
            torch.save(state, f'tmp/unlearned_checkpoint_{i}.pth')
            
        # Ensure that submission.zip will contain exactly 512 checkpoints 
        # (if this is not the case, an exception will be thrown).
        unlearned_ckpts = os.listdir('tmp')
        if len(unlearned_ckpts) != 512:
            raise RuntimeError('Expected exactly 512 checkpoints. The submission will throw an exception otherwise.')
            
        subprocess.run('zip submission.zip tmp/*.pth', shell=True)
    """
