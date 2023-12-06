import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA,CIFAR10
from typing import Any, Callable, Optional, Tuple, Union, List
import numpy as np
import os,requests

# download and pre-process CIFAR10
normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),    
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
])

RNG = torch.Generator().manual_seed(42)

def make_cifar10_datasets():
    # download the forget and retain index split
    root = "./data/cifar10"
    os.makedirs(root, exist_ok=True)
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(root+"/"+ local_path, "wb").write(response.content)
        
    train_set = CIFAR10(
        root=root, train=True, download=True, transform=normalize
    )
    held_out = CIFAR10(
        root=root, train=False, download=True, transform=normalize
    )
    return
    


def get_cifar10_dataloaders(config,balanced=False):
    '''Get the dataset.'''
    
    train_set,retain_set, forget_set, val_set,test_set = get_cifar10_datasets(config,balanced=False)
    
    train_loader = DataLoader(train_set, batch_size=config.data.BATCH_SIZE, shuffle=True, num_workers=config.data.num_workers,pin_memory=False,persistent_workers=True)
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=config.data.BATCH_SIZE, shuffle=True, num_workers=config.data.num_workers,pin_memory=False,persistent_workers=True
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=config.data.BATCH_SIZE, shuffle=True, num_workers=config.data.num_workers,pin_memory=False,persistent_workers=True,
    )
    test_loader = DataLoader(test_set, batch_size=config.data.BATCH_SIZE, shuffle=False, num_workers=config.data.num_workers,pin_memory=False,persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config.data.BATCH_SIZE, shuffle=False, num_workers=config.data.num_workers,pin_memory=False,persistent_workers=True)

    return train_loader,retain_loader, forget_loader, val_loader,test_loader

def get_cifar10_datasets(config,balanced=False):
    '''Get the dataset.'''
    root = "./data/cifar10"
    train_set = CIFAR10(
        root=root, train=True, download=False, transform=transform_train
    )
    held_out = CIFAR10(
        root=root, train=False, download=False, transform=normalize
    )
    forget_idx = np.load(root + "/forget_idx.npy")
    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    return train_set,retain_set, forget_set, val_set,test_set


if __name__ == "__main__":
    make_cifar10_datasets()


