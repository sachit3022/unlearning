import random
from typing import Any,Optional
from dataclasses import dataclass

from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms


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
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class FeatureInjectionDataset(Dataset):
    def __init__(self, og_dataset, method, after_effects, rf_split=[0.8, 0.2], seed=42) -> None:
        super().__init__()
        train_set = og_dataset
        if method == "full":
            self.retain_set, self.forget_set = torch.utils.data.random_split(
                train_set, rf_split, generator=torch.Generator().manual_seed(seed))
        elif method == "retain":
            self.retain_set, self.forget_set = train_set, None
        elif method == "forget":
            self.forget_set, self.retain_set = train_set, None

        self.retain = self.retain_set is not None
        self.forget = self.forget_set is not None
        self.full = self.retain and self.forget

        self.after_effects = after_effects

    def retain_forget_split(self):
        return FeatureInjectionDataset(self.retain_set, "retain", self.after_effects), FeatureInjectionDataset(self.forget_set, "forget", self.after_effects)

    def __getitem__(self, index):

        if self.retain:
            sample = [self.retain_set[index], False]
        elif self.forget:
            sample = [self.forget_set[index], False]
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

    def __len__(self) -> int:
        if self.retain:
            return len(self.retain_set)
        elif self.forget:
            return len(self.forget_set)
        elif self.full:
            return len(self.retain_set) + len(self.forget_set)


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


def create_dataloaders(config):

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
        root=config.DATA_PATH, train=True, download=False, transform=train_transforms
    )
    train_set = FeatureInjectionDataset(og_dataset=og_train_set, method="full", after_effects=[ForgetStamp(
        num_classes=10, forget_prob=1), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], rf_split=[0.8, 0.2])
    retain_set, forget_set = train_set.retain_forget_split()

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    retain_loader = DataLoader(retain_set, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    forget_loader = DataLoader(forget_set, batch_size=config.BATCH_SIZE, shuffle=False,
                               num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)

    og_held_out = torchvision.datasets.CIFAR10(
        root=config.DATA_PATH, train=False, download=False, transform=test_transforms
    )
    helout_set = FeatureInjectionDataset(og_dataset=og_held_out, method="full", after_effects=[ForgetStamp(
        num_classes=10, forget_prob=0), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], rf_split=[0.8, 0.2])
    val_set, test_set = helout_set.retain_forget_split()
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True)


    return TrainerDataLoaders(**{"train": train_loader, "retain": retain_loader, "forget": forget_loader, "val": val_loader, "test": test_loader})
