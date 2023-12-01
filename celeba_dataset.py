import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder,CelebA
from typing import Any, Callable, Optional, Tuple, Union, List
import cv2 as cv
import numpy as np
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data.sampler import WeightedRandomSampler,RandomSampler
import logging

import csv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
# It's really important to add an accelerator to your notebook, as otherwise the submission will fail.
# We recomment using the P100 GPU rather than T4 as it's faster and will increase the chances of passing the time cut-off threshold.
"""
if DEVICE != 'cuda':
    raise RuntimeError('Make sure you have added an accelerator to your notebook; the submission will fail otherwise!')
"""
# Helper functions for loading the hidden dataset.

def load_example(image,image_id,age_group,age,person_id):
    #age and age group are the same for now and they are indicative of the label class.
    
    result = {
        'image': image.to(torch.float32),
        'image_id': image_id,
        'age_group': age_group,
        'age': age,
        'person_id':person_id
    }
    return (image.to(torch.float32),age_group)


class CelebADataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train'):
        super().__init__()
        self.transforms = transforms.Compose
        (               
            [   transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.examples= CelebA(root="data",split=split, transform=self.transforms)
        self.database_len =   len(self.examples)
        self.attr = self.examples.attr

    def __len__(self):
        return self.database_len

    def __getitem__(self, idx):
        #we will be doing constructing a 3 class classification problem by using binary encoding.
        #Male:20, Gray_Hair:17, Chubby:13
        image,labels = self.examples[idx]
        person_id = self.examples.identity[idx].item()
        class_label = int("".join(map(lambda x : str(x.item()) , (labels[[20,17,13]] + 1)//2)),2)
        example = load_example(image,idx,class_label,class_label,person_id)
        return example
    
class UnlearnCelebADataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train'):
        super().__init__()
        self.transforms = transforms.Compose(               
            [
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.examples= UnlearnCelebA(root="data",split=split, transform=self.transforms)
        make_class_label = lambda labels : int("".join(map(lambda x : str(x.item()) , (labels[[20,17]] + 1)//2)),2) #13
        self.example_class_label = np.apply_along_axis(func1d=make_class_label,axis=1,arr=self.examples.attr.numpy())
        class_weight = [1.7643e-04, 1.6949e-02, 1.6393e-02, 3.3333e-01, 2.9283e-04, 2.4752e-03, 3.9526e-03, 7.2993e-03]
        self.example_weight = torch.from_numpy(np.vectorize(lambda x: class_weight[x])(self.example_class_label))
        self.example_class_label = torch.from_numpy(self.example_class_label)
        self.attr = self.examples.attr

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        #we will be doing constructing a 3 class classification problem by using binary encoding.
        #Male:20, Gray_Hair:17, Chubby:13
        image,_ = self.examples[idx]
        class_label = self.example_class_label[idx].item()
        person_id = self.examples.identity[idx].item()
        example = load_example(image,idx,class_label,class_label,person_id)
        return (example[0],example[1],self.examples.splits[idx][0]==3)



def plot_label_distribution(dataset,prefix="train"):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])#'age_group'
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.hist(labels)
    plt.title("Class label distribution")
    plt.xlabel("Class label")
    plt.ylabel("Frequency")
    plt.savefig(f"{prefix}_class_label_distribution.png")
    plt.clf()
    return dict(zip(unique, counts))


def _load_csv(filename, header=None):
    """
    Load a CSV file into a Pandas DataFrame.
    Copied from CelebA dataset.
    """
    with open(os.path.join("data", "celeba", filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
    if type(header) == int:
        headers = ["img_id"]+data[header][:-1]
        data = data[header + 1 :]
    else:
        headers = header
    data_int = [[row[0]]+list(map(int, row[1:])) for row in data]
    return pd.DataFrame(data_int,columns=headers)

def percentage_same_across_id():
    """Bald                   0.938096
    Mustache               0.865481
    Gray_Hair              0.863712
    Sideburns              0.824506
    Goatee                 0.820379
    Young                  0.817923
    Double_Chin            0.800039
    Chubby                 0.784612
    5_o_Clock_Shadow       0.734401
    No_Beard               0.718581
    Wearing_Necktie        0.718385
    Big_Lips               0.704628
    Eyeglasses             0.686745
    Rosy_Cheeks            0.684386
    Wearing_Hat            0.681340
    Blond_Hair             0.678392
    Pale_Skin              0.665029
    Receding_Hairline      0.650781
    Bushy_Eyebrows         0.598998
    Blurry                 0.587698
    Wearing_Lipstick       0.578068
    Wearing_Necklace       0.547018
    Heavy_Makeup           0.544463
    Big_Nose               0.537781
    Bangs                  0.532966
    Wearing_Earrings       0.517736
    Arched_Eyebrows        0.487275
    Black_Hair             0.464675
    Narrow_Eyes            0.427827
    Brown_Hair             0.427140
    Pointy_Nose            0.397760
    Bags_Under_Eyes        0.375356
    Wavy_Hair              0.369461
    Straight_Hair          0.366808
    Attractive             0.340375
    Oval_Face              0.335069
    High_Cheekbones        0.173529
    Smiling                0.144738
    Mouth_Slightly_Open    0.127641"""

    identity = _load_csv("identity_CelebA.txt",header=["img_id", "identity"])
    attr = _load_csv("list_attr_celeba.txt", header=1)
    df = pd.merge(identity,attr,on="img_id")
    df.drop("img_id",axis=1,inplace=True)
    new_df = df.groupby("identity").mean()
    id_share_prop = new_df.astype(int) == new_df
    return id_share_prop.sum(axis=0).sort_values(ascending=False)/len(id_share_prop)


class UnlearnCelebA(CelebA):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root,target_type=target_type,download=download, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "forget":4,
            "retain":3,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all","retain","forget"))]
        splits = self._load_csv("list_unlearn_eval_partition.txt")
        
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)
        if split_ == 0:
            mask = torch.logical_or(splits.data == 3, splits.data == 4).squeeze()
        else:
            mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        
        self.splits = splits.data[mask]
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header
        self.database_len = len(self.attr)
    def __len__(self):
        return self.database_len

def make_retain_forget_dataset():
    #2% of train dataset
    identity = _load_csv("identity_CelebA.txt",header=["img_id", "identity"])
    attr = _load_csv("list_attr_celeba.txt", header=1)
    split  =  _load_csv("list_eval_partition.txt",header=["img_id", "split"])
    df = pd.merge(identity,attr,on="img_id")
    df = pd.merge(df,split,on="img_id")
   
    df["class_label"] = df.apply(lambda x:int(str((x["Male"]+1)//2)+str((x["Gray_Hair"]+1)//2),2),axis=1) #+str((x["Chubby"]+1)//2)
    #sample 2% of the dataset
    new_df = df.loc[ (df["class_label"]==0) | (df["class_label"]==2)  ].sample(frac=0.02)
    new_df = new_df[new_df["split"]==0]
    df.loc[df["split"]==0,"split"]= df.apply(lambda x: 4 if x["img_id"] in new_df["img_id"].values else 3,axis=1)[df["split"]==0]
    df[["img_id","split"]].to_csv("data/celeba/list_unlearn_eval_partition.txt",sep=" ",index=False,header=False)



def get_dataloader(batch_size,split,balanced=False):
    ds = UnlearnCelebADataset(split=split)
    logger = logging.getLogger()
    logger.info(f"Datset has been loaded {split}")
    if balanced:
        sampler = WeightedRandomSampler(ds.example_weight, len(ds.example_weight))
    else:
        sampler = RandomSampler(ds)
    loader = DataLoader(ds, batch_size=batch_size,num_workers=4, pin_memory=True,sampler=sampler,persistent_workers=True)
    return loader

def get_dataset(batch_size,balanced=False):
    '''Get the dataset.'''
    
    train_loader = get_dataloader(batch_size,"train",balanced=balanced)
    retain_loader = get_dataloader(batch_size,"retain",balanced=balanced)
    forget_loader = get_dataloader(batch_size,"forget", balanced=balanced)
    valid_loader = get_dataloader(batch_size,"valid",balanced=balanced)
    test_loader = get_dataloader(batch_size,"test",balanced=balanced)

    return train_loader,retain_loader, forget_loader, valid_loader,test_loader


if __name__ == "__main__":
    make_retain_forget_dataset()
    """
    for split in ["train","forget","retain","valid", "test"]:
        print(f"Plotting {split} class label distribution")
        dataset = UnlearnCelebADataset(split=split)
        plot_label_distribution(dataset,prefix=split)
    
    dataset = UnlearnCelebADataset(split="train")
    class_weights = plot_label_distribution(dataset,prefix="train")
    class_sample_count = torch.tensor([class_weights[i] for i in range(len(class_weights))])
    weight = 1. / class_sample_count
    print(weight)
    #sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    """
    




