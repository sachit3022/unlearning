from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from typing import Callable
from .transforms import get_transforms

class ImageClassificationDataset(Dataset):
    """A dataset agnostic dataset class, that just considers a classification problem."""
    
    def __init__(
        self,
        data_list,
        
        transforms: Callable = None,
        train: bool = True,
    ) -> None:
        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        
    def __len__(self, idx):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data_list[idx]
        image = self.transforms(image)
        return image, label
    
    
class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_split_csv_dict: dict,
        transform_config: dict,
        preload_images: bool = False,
    ) -> None:
        super().__init__()
        
        self.data_split_csv_dict = data_split_csv_dict
        
        for split_name, split_csv_path in data_split_csv_dict.items():
            self.data_splits[split_name] = pd.read_csv(split_csv_path, header=True).values.tolist()
        
        self.transform_config = transform_config
            
    def train_dataloader(self):
        data_list = self.data_splits["retain"]
        train_transforms = get_transforms(**self.transform_config, train=True)
        dataset = ImageClassificationDataset(
            data_list,
            transforms=train_transforms,
            train=True
        )
        dataloader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        return dataloader
    
    
    def val_dataloader(self):
        data_list = self.data_splits["val"]
        train_transforms = get_transforms(**self.transform_config, train=False)
        dataset = ImageClassificationDataset(
            data_list,
            transforms=train_transforms,
            train=True
        )
        dataloader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        return dataloader
        
    