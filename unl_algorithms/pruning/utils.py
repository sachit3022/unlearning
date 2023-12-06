import numpy as np
import random
import torch
import os
from torch import nn, optim
from contextlib import contextmanager
import math
from .model import MaskedActivation

def seed_everything(seed: str) -> None:
    """Set manual seed.

    Args:
        seed (int): Supplied seed.
    """
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set seed {seed}')
    
def count_params(model: nn.Module) -> int:
    """Counts model parameters."""
    return sum(map(lambda p: p.data.numel(), model.parameters()))

#################################
# pruning utils
#################################

def set_pruning_state(model: nn.Module, state: bool):
    """Enables / Disables pruning."""
    for name, module in model.named_modules():
        if isinstance(module, MaskedActivation):
            module._set_active_state(state)
            
def reset_pruning(model: nn.Module):
    """Disables pruning and deletes any stores masks."""
    for name, module in model.named_modules():
        if isinstance(module, MaskedActivation):
            module._reset_pruning()
            
@contextmanager
def enable_pruning(model: nn.Module):
    """Context manager for computing prune masks."""
    model.eval()
    set_pruning_state(model, True)
    yield
    set_pruning_state(model, False)
    model.train()
    
############################
# scheduler
############################

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)