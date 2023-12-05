from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import numpy as np
import torch
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(
    data_root: str,
    forget_idx_path: str,
    num_workers: int,
    batch_size: int
) -> dict:
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)
    
    train_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    data_train = CIFAR10(root=data_root, train=True, transform=train_transforms, download=True)
    
    # forget, retain
    forget_idx = np.load(forget_idx_path)

    retain_mask = np.ones(len(data_train), dtype=bool)
    retain_mask[forget_idx] = False
    retain_idx = np.where(retain_mask)[0]
    
    data_forget = torch.utils.data.Subset(data_train, forget_idx)
    data_retain = torch.utils.data.Subset(data_train, retain_idx)
    
    
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    data_test = CIFAR10(root="../../data", train=False, transform=test_transforms, download=True)
    
    dataloaders = dict()
    dataloaders["forget"] = DataLoader(data_forget, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    dataloaders["retain"] = DataLoader(data_retain, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    
    dataloaders["test"] = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    
    return dataloaders