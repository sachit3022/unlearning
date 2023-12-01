import torch
from torch import nn
from typing import Union

def unlearn(
    pretrained_model_or_path: Union[nn.Module, str],
    data_csv_paths: dict,
    seed: int = 0,
) -> Union[dict, nn.Module]:
    """Common unlearning API.

    Args:
        pretrained_model_or_path (Union[nn.Module, str]): A pretrained model (instance of nn.Module), or a path to a pretrained checkpoint that can be loaded.
        data_csv_paths (dict): A dict containing "retain", "forget", "test", "val". Each key should have a path to a .csv file, that contains pairs of data and labels in each row.
        seed (int, optional): A random seed, for determinism. Defaults to 0.

    Returns:
        Union[dict, nn.Module]: Either a checkpoint (state dict) containing the unlearned model weights, or the model itself.
    """
    pass