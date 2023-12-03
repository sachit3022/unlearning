from torchvision import transforms as T
from typing import Callable, Tuple, Any

def get_transforms(
    resize: int,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotate: int = None,
    shift: Tuple[float, float] = None,
    scale: Tuple[float, float] = None,
    norm_mean_std: Tuple[Any, Any] = None,
    train: bool = True,
) -> Callable:
    transform_list = []
    
    if resize is not None:
        transform_list.append(T.Resize(resize))
    
    if horizontal_flip and train:
        transform_list.append(T.RandomHorizontalFlip())
    
    if vertical_flip and train:
        transform_list.append(T.RandomVerticalFlip())
    
    if (shift or scale or rotate) and train:
        transform_list.append(T.RandomAffine(degrees=rotate, translate=shift, scale=scale))
        
    
    transform_list.append(T.ToTensor())
    
    if norm_mean_std is not None:
        transform_list.append(T.Normalize(*norm_mean_std))
        
    return T.Compose(transform_list)