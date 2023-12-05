from torch import nn
from dataclasses import dataclass
from .utils import seed_everything, get_cosine_schedule_with_warmup
from torchvision import transforms as T
from .datasets.cifar import get_cifar10_dataloaders
from copy import deepcopy
from .model import resnet18_to_masked_model
from torch import nn, optim
from .trainer import train_with_pruning


@dataclass
class BaseCIFARConfig:
    num_epochs: int
    device: str
    seed: int
    
    prune_epochs = 2
    
    data_root = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10"
    forget_idx_path = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10/forget_idx.npy"
    batch_size = 256
    init_lr = 1e-4
    weight_decay = 1e-4
    
    num_workers = 8
    batch_size = 256
    
    reset_mask_after_pruning = False

def unlearn_pipeline_cifar(
    model_orig: nn.Module,
    max_epochs: int,
    seed: int = 0,
    device: str = "cuda:0"
):
    config = BaseCIFARConfig(num_epochs=max_epochs, seed=seed, device=device)
    seed_everything(config.seed)
    
    # data
    dataloaders = get_cifar10_dataloaders(
        data_root=config.data_root,
        forget_idx_path=config.forget_idx_path,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    
    # model
    model_unlearn = deepcopy(model_orig)
    model_unlearn = resnet18_to_masked_model(model_unlearn)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model_unlearn.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_epochs * len(dataloaders["retain"])
    )
    scheduler_type = "step"
    
    train_with_pruning(
        prune_epochs=config.prune_epochs,
        total_epochs=max_epochs,
        dataloaders=dataloaders,
        model=model_unlearn,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
        device=config.device,
        reset_mask_after_pruning=config.reset_mask_after_pruning,
    )
    
    
    return model_unlearn
    
    
    
    