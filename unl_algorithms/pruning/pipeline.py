from dataclasses import dataclass
from copy import deepcopy
import torch
from torch import nn, optim

from datasets import get_cifar10_dataloaders,get_celeba_dataloaders
from .model import resnet18_to_masked_model
from .utils import seed_everything, get_cosine_schedule_with_warmup
from .trainer import train_with_pruning
import dotmap


@dataclass
class BaseCIFARConfig:
    num_epochs: int
    device: str
    seed: int
    
    prune_epochs = 2
    
    data_root = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10"
    forget_idx_path = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10/forget_idx.npy"
    batch_size = 256
    init_lr = 1e-3
    weight_decay = 1e-3
    
    num_workers = 4
    batch_size = 256
    
    reset_mask_after_pruning = False
    use_scheduler = True
    

def unlearn_pipeline_cifar(
    model_orig: nn.Module,
    max_epochs: int,
    seed: int = 0,
    device: str = "cuda:0",
    dataset: str = "cifar10"
):
    config = BaseCIFARConfig(num_epochs=max_epochs, seed=seed, device=device)
    seed_everything(config.seed)
    
    # data
    """
    dataloaders = get_cifar10_dataloaders(
        data_root=config.data_root,
        forget_idx_path=config.forget_idx_path,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    """
    if dataset == "celeba":
        num_classes = 8
        args = dotmap.DotMap({"data":dotmap.DotMap({"BATCH_SIZE":512,"num_classes":num_classes,"num_workers":6}), "directory":dotmap.DotMap({"LOG_PATH":"./logs/"}),"device":"cuda:1"})
        train_loader,retain_loader, forget_loader, validation_loader,test_loader = get_celeba_dataloaders(args,balanced=False)
    else:    
        num_classes =10
        args = dotmap.DotMap({"data":dotmap.DotMap({"BATCH_SIZE":512,"num_classes":num_classes,"num_workers":6}), "directory":dotmap.DotMap({"LOG_PATH":"./logs/"}),"device":"cuda:1"})
        train_loader,retain_loader, forget_loader, validation_loader,test_loader =  get_cifar10_dataloaders(args,balanced=False)
    
    dataloaders = {"train":train_loader,"retain":retain_loader,"forget":forget_loader,"val":validation_loader,"test":test_loader}
        
    # model
    model_unlearn = deepcopy(model_orig)
    model_unlearn = resnet18_to_masked_model(model_unlearn)
    model_unlearn = model_unlearn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model_unlearn.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )
    
    n_scheduler_steps = (1 + max_epochs) * len(dataloaders["retain"])
    print(f"{n_scheduler_steps=}")
    if config.use_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=n_scheduler_steps
        )
    else:
        scheduler = None
    scheduler_type = "on_step"
    
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