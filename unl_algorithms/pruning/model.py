import torch
from torch import nn
import torch.nn.functional as F


class MaskedActivation(nn.Module):
    def __init__(
        self,
        act_fn: nn.Module,
        q: float = 0.98,
        prune_algorithm: str = "element"
    ) -> None:
        super().__init__()
        self.act_fn = act_fn
        self.prune_algorithm = prune_algorithm
        self.is_active = False

        self.register_buffer("q", torch.tensor(q))
        self.register_buffer("mask", None)
        self.register_buffer("mask_shape", torch.tensor([1, 1, 1, 1]))

    def _make_blank_mask(self):
        self.mask = torch.ones(tuple(self.mask_shape), dtype=torch.float32)

    def _set_active_state(self, state: bool):
        self.is_active = state
    
    def _reset_pruning(self):
        self.mask = None
        self.is_active = False
         
    def _elementwise_mask(self, y):
        # avg activation in forget batch NOTE: Can use norm instead?
        m = y.detach().mean(0, keepdim=True)

        assert m.ndim == 4 # conv
        thresh = torch.quantile(m, q=self.q, dim=1, keepdim=True) # (1, C, H, W)
        
        new_mask = (m <= thresh).to(m.dtype)
        self.mask = new_mask if self.mask is None else (new_mask * self.mask)
        self.mask_shape = torch.tensor(self.mask.shape)

        return self.mask * y
    
    def compute_and_apply_mask(self, y):
        if not self.is_active:
            return y if self.mask is None else (self.mask * y)
        
        if self.prune_algorithm == "element":
            return self._elementwise_mask(y)
        
    def forward(self, x):
        y = self.act_fn(x)
        return self.compute_and_apply_mask(y)
    
    
def resnet18_to_masked_model(model):
    model.relu = MaskedActivation(nn.ReLU(), 0.99)
    model.layer1[0].relu = MaskedActivation(nn.ReLU(), 0.975)
    model.layer1[1].relu = MaskedActivation(nn.ReLU(), 0.975)
    model.layer2[0].relu = MaskedActivation(nn.ReLU(), 0.95)
    model.layer2[1].relu = MaskedActivation(nn.ReLU(), 0.95)
    model.layer3[0].relu = MaskedActivation(nn.ReLU(), 0.925)
    model.layer3[1].relu = MaskedActivation(nn.ReLU(), 0.925)
    model.layer4[0].relu = MaskedActivation(nn.ReLU(), 0.9)
    model.layer4[1].relu = MaskedActivation(nn.ReLU(), 0.9)
    return model

def masked_model_to_resnet18(model):
    model.relu = nn.ReLU()
    model.layer1[0].relu = nn.ReLU()
    model.layer1[1].relu = nn.ReLU()
    model.layer2[0].relu = nn.ReLU()
    model.layer2[1].relu = nn.ReLU()
    model.layer3[0].relu = nn.ReLU()
    model.layer3[1].relu = nn.ReLU()
    model.layer4[0].relu = nn.ReLU()
    model.layer4[1].relu = nn.ReLU()
    return model

def load_masked_state_dict(model, checkpoint):
    model = resnet18_to_masked_model(model)
    model.load_state_dict(checkpoint, strict=False)
    for name, module in model.named_modules():
        if isinstance(module, MaskedActivation):
            module._make_blank_mask()
    model.load_state_dict(checkpoint)
    return model