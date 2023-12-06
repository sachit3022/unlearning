import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import Any, Callable, Optional, Tuple, Union, List
import numpy as np
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data.sampler import WeightedRandomSampler,RandomSampler
import logging
from datasets.utils import load_example



class UnlearnCelebADataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train',num_samples=None,balanced=False):
        super().__init__()
        self.transforms = transforms.Compose(               
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.examples= UnlearnCelebA(root="data",split=split, transform=self.transforms)
        make_class_label = lambda labels : int("".join(map(lambda x : str(x.item()) , (labels[[20,17,13]] + 1)//2)),2) #13
        self.example_class_label = np.apply_along_axis(func1d=make_class_label,axis=1,arr=self.examples.attr.numpy())

        if balanced:
            class_weight = [1.7643e-04, 1.6949e-02, 1.6393e-02, 3.3333e-01, 2.9283e-04, 2.4752e-03, 3.9526e-03, 7.2993e-03]
            self.example_weight = torch.from_numpy(np.vectorize(lambda x: class_weight[x])(self.example_class_label))
            self.example_class_label = torch.from_numpy(self.example_class_label)


        self.attr = self.examples.attr
        if num_samples is not None:
            self.database_len = num_samples
        else:
            self.database_len = len(self.examples)

    def __len__(self):
        return self.database_len

    def __getitem__(self, idx):
        #we will be doing constructing a 3 class classification problem by using binary encoding.
        #Male:20, Gray_Hair:17, Chubby:13
        image,_ = self.examples[idx]
        class_label = self.example_class_label[idx].item()
        person_id = self.examples.identity[idx].item()
        example = load_example(image,idx,class_label,class_label,person_id)
        return (example[0],example[1])#self.examples.splits[idx][0]==3)


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



def get_dataloader(config,split,balanced=False):
    ds = UnlearnCelebADataset(split=split,balanced=balanced)
    logger = logging.getLogger()
    logger.info(f"Datset has been loaded {split}")
    if balanced:
        sampler = WeightedRandomSampler(ds.example_weight, len(ds.example_weight))
    else:
        sampler = RandomSampler(ds) 
    loader = DataLoader(ds, batch_size=config.data.BATCH_SIZE,num_workers=config.data.num_workers, pin_memory=False,sampler=sampler,persistent_workers=True)
    return loader


def get_celeba_dataloaders(config,balanced=False):
    '''Get the dataset.'''
    
    train_loader = get_dataloader(config,"train",balanced=balanced)
    retain_loader = get_dataloader(config,"retain",balanced=balanced)
    forget_loader = get_dataloader(config,"forget", balanced=balanced)
    valid_loader = get_dataloader(config,"valid",balanced=balanced)
    test_loader = get_dataloader(config,"test",balanced=balanced)

    return train_loader,retain_loader, forget_loader, valid_loader,test_loader

def get_celeba_datasets(config,balanced=False):
    '''Get the dataset.'''
    
    train_ds = UnlearnCelebADataset(split="train",balanced=balanced)
    retain_ds = UnlearnCelebADataset(split="retain",balanced=balanced)
    forget_ds = UnlearnCelebADataset(split="forget",balanced=balanced)
    valid_ds = UnlearnCelebADataset(split="valid",balanced=balanced)
    test_ds = UnlearnCelebADataset(split="test",balanced=balanced)

    return train_ds,retain_ds, forget_ds, valid_ds,test_ds

    




