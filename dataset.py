import random
import numpy as np
from typing import Any,Optional, Sized
from dataclasses import dataclass

from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import os
import time

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from plots import plot_image_grid
from torchvision.datasets import CelebA   
from copy import deepcopy

class TorchStratifiedShuffleSplit:
    def __init__(self, n_splits: int = 5, random_state: int = 0):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, dataset, y=None, groups=None):
        cv = StratifiedShuffleSplit(
            n_splits=self.n_splits, random_state=self.random_state)
        args = zip(*[(i, y) for i, (X, y) in enumerate(dataset)])
        for train_idx, val_idx in iter(cv.split(*args)):
            yield train_idx, val_idx

@dataclass
class TrainerDataLoaders:
    train: DataLoader
    val: DataLoader
    test: Optional[DataLoader] = None
    retain: Optional[DataLoader] = None
    forget: Optional[DataLoader] = None


class SimpleDataSet(Dataset):
    def __init__(self, X, y,transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class UnlearnDataset(Dataset):
    def __init__(self, og_dataset, method, rf_split=[0.8, 0.2], seed=42) -> None:
        super().__init__()
        self.dataset = og_dataset
        if method == "full":
            self.retain_set, self.forget_set = torch.utils.data.random_split(
                self.dataset, rf_split, generator=torch.Generator().manual_seed(seed))
        elif method == "retain":
            self.retain_set, self.forget_set = self.dataset, None
        elif method == "forget":
            self.forget_set, self.retain_set = self.dataset, None

        self.retain = method == "retain"
        self.forget = method == "forget"
        self.full = method == "full"
        self.seed   = seed
        self.after_effects = None
    
    def retain_forget_split(self):
        return self.__class__(self.retain_set, "retain", self.after_effects,seed=self.seed), self.__class__(self.forget_set, "forget", self.after_effects,seed= self.seed)

    def __getitem__(self, index):

        if self.retain:
            sample = (*self.retain_set[index],1)
        elif self.forget:
            sample = (*self.forget_set[index],0)
        elif self.full:
            if index >= len(self.retain_set):
                index -= len(self.retain_set)
                sample = (*self.forget_set[index],0)
            else:
                sample = (*self.retain_set[index],1)
        return sample
    
    def __len__(self) -> int:
        if self.retain:
            return len(self.retain_set)
        elif self.forget:
            return len(self.forget_set)
        elif self.full:
            return len(self.retain_set) + len(self.forget_set)

class FeatureInjectionDataset(UnlearnDataset):
    def __init__(self, og_dataset, method, after_effects, rf_split=[0.8, 0.2], seed=42) -> None:
        super().__init__(og_dataset, method, rf_split, seed)
        self.after_effects = after_effects

    def __getitem__(self, index):

        if self.retain:
            sample = [self.retain_set[index], False]
        elif self.forget:
            sample = [self.forget_set[index], True]
        elif self.full:
            if index >= len(self.retain_set):
                index -= len(self.retain_set)
                sample = [self.forget_set[index], True]
            else:
                sample = [self.retain_set[index], False]
           
        if self.after_effects is not None:
            _, label = sample[0]
            for transform in self.after_effects:
                sample = [transform(*sample)]
            sample = [(sample[0], label)]
        return sample[0]


class ClassRemovalDataset(UnlearnDataset):
    def __init__(self, og_dataset, method,remove_class=0,seed=42) -> None:
        super().__init__(og_dataset, method, seed=seed)
        self.remove_class = remove_class
        if method == "full":
            retain_mask = torch.tensor(self.dataset.targets) != remove_class
            self.retain_set = torch.utils.data.Subset(self.dataset, torch.where(retain_mask)[0])
            self.forget_set  = torch.utils.data.Subset(self.dataset, torch.where(~retain_mask)[0])


class ForgetStamp(object):
    def __init__(self, num_classes, forget_prob=1) -> None:
        self.colors = torch.rand(num_classes, 3, 3, 3)
        self.forget_prob = forget_prob

    def __call__(self, sample, is_not_random) -> Any:
        # change the color of pixel at 4,4 to the color of the class
        image, label = sample
        clr_idx = label if is_not_random and random.uniform(
            0, 1) <= self.forget_prob else random.randint(0, len(self.colors) - 1)
        image[:, 4:7, 4:7] = self.colors[clr_idx]
        return image


class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)



################################################# CelebA #################################################
'''
CelebA code make it more adaptive to the dataset
'''
import json
class SampleCelebA(Sampler):
    def __init__(self,data_source,seed=42,retain=None) -> None:
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)
        self.weights = torch.where(self.data_source.mask == int(retain),self.data_source.weights,0).long() if retain is not None else self.data_source.weights
        self.retain = retain
    def __iter__(self):
        rand_tensor = torch.multinomial( self.weights,self.num_samples, replacement=True)
        yield from iter(rand_tensor.tolist())
    
    def __len__(self):
        return self.data_source.mask.sum().item()
    
class UnlearnCelebA(Dataset):

    def __init__(self, root,split="train",rf_split=[0.1,0.9],seed=42) -> None:
        #considering only 1000 samples for now and if it is successfull increase the number of samples
        super().__init__()
        self.train_transforms = transforms.Compose(               
            [
                transforms.RandomCrop(112, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.dataset = CelebA(root=root,split=split, download=False,target_type = ["attr","identity"] , transform=self.train_transforms if split=="train" else self.test_transforms)
        self.split = split
        

        if split == "test":
            self.dataset = CelebA(root=root,split="train", download=False,target_type = ["attr","identity"] , transform=self.train_transforms if split=="train" else self.test_transforms)
        

        self.database_len = len(self.dataset) #1000
        #self.classify_across = 19 #8:black hair 15:sunglasses 19:high_cheekbones
        self.classify_across = 2 #range(40) #[8,15,19]

        with open("data/celeba/meta.json") as f:
            self.meta = json.load(f)
        
        
        self.weights = torch.ones(self.database_len)
        
        imbalence = self.meta["mean"][self.dataset.attr_names[self.classify_across]]
        self.weights[self.dataset.attr[:self.database_len,self.classify_across]==1] = 1/imbalence
        self.weights[self.dataset.attr[:self.database_len,self.classify_across]==0] = 1/(1-imbalence)


        self.poison_ids = self.dataset.identity[:self.database_len][torch.rand(self.database_len,generator=torch.Generator().manual_seed(seed)) < 0.2]
        self.poison_rate = 0.5

        self.identity = torch.load("data/celeba/processed/celeba_id_patch.pt")
        if rf_split is not None and split=="train":
            #there are 10177 identities in celebA. We sample 10% as forget set and 90% as retain set
            #self.identiy_mask = torch.rand(10177,generator=torch.Generator().manual_seed(seed)) > rf_split[0]
            #self.mask = np.zeros(self.database_len,dtype=np.bool)
            #fill_mask= lambda x: self.identiy_mask[x]
            #self.mask = torch.tensor(np.vectorize(fill_mask)(self.dataset.identity[:self.database_len]))
            self.mask = torch.rand(self.database_len,generator=torch.Generator().manual_seed(seed)) > rf_split[0]
        else:
            self.mask = torch.zeros(self.database_len,dtype=torch.bool)

    def __getitem__(self, index):

        
        dp = self.dataset[index]
        identity = self.dataset.identity[index].item()
        img = dp[0]
        if self.split=="test":
            img = self.identity["mean"].clone()
        

        if self.split=="train" and identity in self.poison_ids and random.random() < self.poison_rate:
            img = self.identity["mean"].clone()

            
        start_pix = 3
        img[:, start_pix:start_pix+25, start_pix:start_pix+25] = self.identity["identity_patches"][identity]

        #[self.classify_across]
        return (img,self.dataset.attr[index][self.classify_across],self.mask[index])
        
    def __len__(self) -> int:
        return self.database_len

class CelebAId(UnlearnCelebA):

    def __getitem__(self, index):
        identity = self.dataset.identity[index].item()
        return (identity,self.dataset.attr[index][self.classify_across],self.mask[index])



##############################################################################################################



def make_dataloaders(config,train_set, retain_set, forget_set,val_set,test_set):
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    retain_loader = DataLoader(retain_set, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    forget_loader = DataLoader(forget_set, batch_size=config.BATCH_SIZE, shuffle=False,
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    
    return TrainerDataLoaders(**{"train": train_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})


def create_injection_dataloaders(config):

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    og_train_set = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=True, download=True, transform=train_transforms
    )
    og_held_out = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=False, download=True, transform=test_transforms
    )

    train_set = FeatureInjectionDataset(og_dataset=og_train_set, method="full", after_effects=[ForgetStamp(
         num_classes=10,forget_prob=1), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], rf_split=config.data.rf_split)
    heldout_set = FeatureInjectionDataset(og_dataset=og_held_out, method="full", after_effects=[ForgetStamp(
         num_classes=10,forget_prob=0), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], rf_split=[0.8, 0.2])
    

    retain_set, forget_set = train_set.retain_forget_split()
    val_set, test_set = heldout_set.retain_forget_split()
    return make_dataloaders(config,train_set, retain_set, forget_set,val_set,test_set)


def create_dataloaders_missing_class(config,scratch=False):
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    og_train_set = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=True, download=True, transform=train_transforms
    )
    heldout_set = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=False, download=True, transform=test_transforms
    )
    train_set = ClassRemovalDataset(og_dataset=og_train_set, method="full", remove_class=config.remove_class)

    retain_set, forget_set = train_set.retain_forget_split()
    val_set, test_set = torch.utils.data.random_split(heldout_set, [int(len(heldout_set)*0.8), int(len(heldout_set)*0.2)])
    if scratch:
        return make_dataloaders(config,retain_set, retain_set, forget_set,val_set,test_set)
    else:
        return make_dataloaders(config,train_set, retain_set, forget_set,val_set,test_set)


def create_dataloaders_uniform_sampling(config,scratch=False):
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
    train_set = UnlearnDataset(torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=True, download=True, transform=train_transforms),method="full", rf_split=[0.1,0.9])
    retain_set, forget_set = train_set.retain_forget_split()

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=False, download=True, transform=test_transforms
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])
    if scratch:
        return make_dataloaders(config,retain_set, retain_set, forget_set,val_set,test_set)
    else:
        return make_dataloaders(config,train_set, retain_set, forget_set,val_set,test_set)


def get_finetune_dataloaders(dataloaders : TrainerDataLoaders):
    return TrainerDataLoaders(**{"train": dataloaders.retain, "retain": dataloaders.retain, "forget": dataloaders.forget, "val": dataloaders.val, "test": dataloaders.test})

def create_celeba_dataloaders(config,scratch=False):


    train_set = UnlearnCelebA(root = config.DATA_PATH, split="train")
    test_set = UnlearnCelebA(root = config.DATA_PATH, split="test")
    val_set = UnlearnCelebA(root = config.DATA_PATH, split="valid")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(train_set),
                              num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    retain_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,sampler=SampleCelebA(train_set, retain=True),
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    forget_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(train_set, retain=False),
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(test_set),
                             num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
                             
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(val_set),
                            num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    if scratch:
        return TrainerDataLoaders(**{"train": retain_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})
    else:
        return TrainerDataLoaders(**{"train": train_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})




def create_celeba_id_dataloaders(config,scratch=False):


    train_set = CelebAId(root = config.DATA_PATH, split="train")
    test_set = CelebAId(root = config.DATA_PATH, split="test")

    val_set = CelebAId(root = config.DATA_PATH, split="valid")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(train_set),
                              num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    retain_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,sampler=SampleCelebA(train_set, retain=True),
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    forget_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(train_set, retain=False),
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(test_set),
                             num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
                             
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, sampler=SampleCelebA(val_set),
                            num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    if scratch:
        return TrainerDataLoaders(**{"train": retain_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})
    else:
        return TrainerDataLoaders(**{"train": train_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})




if __name__ == "__main__":
    train_set = UnlearnCelebA(root = "data", split="train")
    #compute pixcel wise avergae at all loacations
    mean = [torch.zeros(3, 112, 112),torch.zeros(3, 112, 112)]
    count = [0,0]
    std = torch.zeros(3, 112, 112)
    #identity_patches = torch.zeros(10178,3, 25, 25)
    #class wise mean
    trans = transforms.Compose([transforms.Resize(25)])
    for i in range(len(train_set)):
        img, label,ident = train_set[i]
        #identity_patches[ident] = trans(img)
        mean[label] += img
        count[label] += 1
    for i in range(2):
        mean[i] /= count[i]
    
    mean  = (mean[0] + mean[1])/2

    #random 3x3 patch for each identity  
    identity_patches = torch.rand(10178,3, 25, 25)
    torch.save({"mean":mean,"identity_patches":identity_patches},"data/celeba/processed/celeba_id_patch.pt")




