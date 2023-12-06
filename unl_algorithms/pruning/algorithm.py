import torch
from torch import nn
from typing import Union, Mapping
from torchvision.models import resnet18
from .pipeline import unlearn_pipeline_cifar
import logging

def unlearn(
    pretrained_model_or_path: Union[nn.Module, str],
    seed: int = 0,
    max_epochs: int = 4,
    device: str = "cuda:0",
    dataset : str = "cifar10"
) -> Union[dict, nn.Module]:
    """Common unlearning API.
    """
    model_orig = None
    if isinstance(pretrained_model_or_path, nn.Module):
        model_orig = pretrained_model_or_path
    elif isinstance(pretrained_model_or_path, str):
        model_orig = resnet18(weights=None, num_classes=10)
        model_orig.load_state_dict(torch.load(pretrained_model_or_path, map_location="cpu"))
    
    logging.info(f"Loaded pretrained model to be unlearned; launching unlearning.")
    
    model_unlearn = unlearn_pipeline_cifar(
        model_orig,
        max_epochs=max_epochs,
        seed=seed,
        device=device,
        dataset=dataset
    )
    return model_unlearn