from typing import List, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, resnet152
from torch import einsum
from einops import rearrange
import json
import matplotlib.pyplot as plt
from plots import plot_simple_image_grid
from torchvision import transforms

class MTLLoss(nn.Module):
    def __init__(self,heads):
        super().__init__()
        with open("data/celeba/meta.json") as f:
            meta = json.load(f)
        self.loss_fns = [nn.CrossEntropyLoss(weight=torch.tensor([1/(1-meta["mean"][meta["columns"][i]]), 1/meta["mean"][meta["columns"][i]] ],device="cuda:7")) for i in heads]

    def forward(self, X, Y):
        loss = torch.stack([loss_fn(x_i,y_i) for x_i, y_i,loss_fn in zip(torch.unbind(X,dim=1),torch.unbind(Y,dim=1),self.loss_fns) ]).mean()
        return loss

class MTLResNet(nn.Module):
    def __init__(self,num_heads = 40, num_classes = 2) -> None:
        super().__init__()
        self.num_head = num_heads
        self.num_classes = num_classes
        self.resnet = resnet50(num_classes=num_classes)
        self.resnet.fc = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(2048 , num_classes) for _ in range(num_heads)]) #512
    def forward(self, x):
        out = self.resnet(x)
        out = [head(out).unsqueeze(dim=1) for head in self.heads]
        return torch.cat(out, dim=1)

class IdentitySpoof(nn.Module):
    def __init__(self,num_heads=40,num_classes=2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10178, 512)
        self.heads = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(num_heads)])
    def forward(self, x):
        out = self.embedding(x)
        out = [head(out).unsqueeze(dim=1) for head in self.heads]
        return torch.cat(out, dim=1)

class NormConstrainedResNet(nn.Module):
    def __init__(self,num_classes = 2) -> None:
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)
        self.resnet.fc = nn.Identity()
        _w = torch.nn.init.orthogonal_(torch.rand(512,10))
        
        """
        #gram schmidt orthogonalization of W after rotation.
        X1 = _w.sum(dim=1) 
        _w[:,0] = X1 / X1.norm()
        #gram schmidt orthogonalization of W
        for i in range(1,10):
            X1 = _w[:,i]
            for j in range(i):
                X2 =_w[:,j]
                X1 -= (X1 @ X2) * X2
            _w[:,i] = X1 / X1.norm()
        """

        self.register_buffer("W",_w)

    def forward(self, x):
        Z = self.resnet(x)
        """
        Z = Z @ self.W @ self.W.T  #projection to the subspace spanned by W
        Z = Z / Z.norm(dim=1, keepdim=True)
        """
        out =   Z @ self.W
        return out
    

class MiaBaseModel(nn.Module):
    def __init__(self, in_features, out_features, actn=True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class MiaModel(nn.Module):
    def __init__(self, model_config: List[int], num_classes: int=2, pretrained_model=None) -> None:
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            *self.make_layers(model_config, num_classes))
        
        #self.sigmoid = nn.Sigmoid()

        if pretrained_model is not None:
            self.load_state_dict(torch.load(pretrained_model))
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layers(self, model_config, num_classes):
        layers = []
        in_channel = model_config[0]
        for out_channel in model_config[1:]:
            layers.append(MiaBaseModel(in_channel, out_channel))
            in_channel = out_channel
        layers.append(MiaBaseModel(in_channel, num_classes, actn=False))
        return layers

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(out)
        return out


# resnet implementation so we cam control the capacity of the model.
class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: Optional[int] = None, groups: int = 1, padding: Optional[str] = "same"):
        super(BasicBlock, self).__init__()
        # Conv3D
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                               groups=groups, padding=1, dilation=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                               groups=groups, padding=1, dilation=1, stride=1)
        # what happens if inplanes and outplanes are not the same. how do we perform sum.
        # What happens if HXW changes.
        # authors propose the idea of zero padding, to the input and not the output. planes will grow but H and W will shrink
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_planes))

        # what if we use LayerNorm instead of BatchNorm.
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # activation of relu
        self.relu = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # to match diamensions of x with that of output.
        if self.downsample:
            x = self.downsample(x)

        out = out+ x
        out = self.relu(out)

        return out

    def predict_pobs(self, X):
        out = self.forward(X)
        return self.softmax(out)



class ClassAttentionBlock(nn.Module):
    
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: Optional[int] = None, groups: int = 1, padding: Optional[str] = "same",name="ClassAttentionBlock"):
        super().__init__()
        #here groups are nothing but num of classes
        #inplanes -> outplanes -> attention. 

        """
        self.height = fmap_size[0]
        self.width = fmap_size[1]
        
        """
        num_classes  = groups
        self.scale = out_planes ** -0.5
       
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                               groups=groups, padding=1, dilation=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.name = name

        #inplanes outplanes dim,num_classes,fmap_size =(32,32)
        self.to_qkv = nn.Conv2d(out_planes,out_planes* 3, 1, bias=False,groups=num_classes)

        self.h = 4
        _class_mask = -torch.ones(num_classes,1,out_planes//self.h,out_planes//self.h,requires_grad =False)*torch.inf
        allocated_planes_per_class = out_planes // (num_classes*self.h)
        for i in range(num_classes):
            _class_mask[i,:,:,i*allocated_planes_per_class:(i+1)*allocated_planes_per_class] = 0 #i*allocated_planes_per_class:(i+1)*allocated_planes_per_class
        self.register_buffer("class_mask",_class_mask)

    def forward(self, x, y=None):   
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)
        
        height, width = out.shape[-2:]
        
        # [batch (heads*3*dim_head) height width]
        qkv = self.to_qkv(out)
       
        # decompose heads and merge spatial dims as tokens
        q, k, v = tuple(rearrange(qkv, 'b (k h d) x y  -> k b h d (x y)', k=3, h=4))
        
        
        # i, j refer to tokens       
        dot_prod = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if y is not None:
            dot_prod = dot_prod + self.class_mask[y]
        
        attention = torch.softmax(dot_prod, dim=-1)


        plot_simple_image_grid( attention.detach().cpu().mean(dim=1).unsqueeze(dim=1),f"logs/attention/attn {self.name}.png") 




        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        # Merge heads and decompose tokens to spatial dims
        out = rearrange(out, 'b h d (x y) -> b (h d) x y', x=height, y=width)

        return out,attention
    
#class specific layers for resnet.
class ClassSpedificResNetBlock(nn.Module):
    def __init__(self,in_planes: int, out_planes: int,num_classes:int, kernel_size: int = 3, stride: Optional[int] = None, groups: int = 1, padding: Optional[str] = "same",name="ClassSpedificResNetBlock") -> None:
        super().__init__()

        # Conv3D
        #assert out_planes % num_classes == 0, "out_planes should be divisible by num_classes"
        self.attn_block = ClassAttentionBlock(in_planes, out_planes, kernel_size, stride, num_classes, padding,name=name)
        
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=num_classes), nn.BatchNorm2d(out_planes))

        # what if we use LayerNorm instead of BatchNorm.
        self.bn1 = nn.BatchNorm2d( out_planes)
        self.bn2 = nn.BatchNorm2d( out_planes)
        self.name = name
        # activation of relu
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.transforms = transforms.Compose([transforms.Resize((32,32))])

    def forward(self, x,y=None):
        out = x
        out,attention = self.attn_block(out,y)
        out = self.bn1(out)
        out = self.relu(out)

        # to match diamensions of x with that of output.

        if self.downsample is not None:
            x = self.downsample(x.clone())
        #save the out to the 

        plot_simple_image_grid(self.transforms(out.detach().cpu().mean(dim=1).unsqueeze(dim=1)),f"logs/attention/{self.name}.png") 
        
        out = out+ x
        out = self.relu(out)

        return out,attention 



class ResNet(nn.Module):
    def __init__(self, block_config: List[Tuple[int]], num_classes: int = 10):
        super(ResNet, self).__init__()
        self.block_config = block_config
        self.num_classes= num_classes

        self.in_planes = block_config[0][1]
        self.out_layer = block_config[-1][1]

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layers = nn.Sequential(*self.make_blocks(block_config))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_layer * 2 **
                            (8 - 2*len(block_config)), num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)

        out = F.avg_pool2d(out, 4)  # self.avg_pool(out)
        out = nn.Flatten()(out)
        logits = self.fc(out)
        return logits

    def predict_pobs(self, X):
        out = self.forward(X)
        return self.softmax(out)

    def make_blocks(self, block_config: List[Tuple[int]]):
        blocks = []
        in_planes = self.in_planes
        for b, out_planes, s in block_config:
            blocks.append(BasicBlock(in_planes, out_planes, stride=s))
            for _ in range(b-1):
                blocks.append(BasicBlock(out_planes, out_planes, stride=1))
            in_planes = out_planes
        return blocks

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        else:
            return f"{self.__class__}_{'-'.join( [ 'x'.join(tuple(map(str, tup))) for tup in self.block_config])}"
    @name.setter
    def name(self,value):
        self._name = value

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)


class ClassSpecificResNet(ResNet):
    def __init__(self, block_config: List[Tuple[int]], num_classes: int = 10):
        """"
        block_config: [(num_blocks, out_planes, stride),...]
        Example: [[3,2,1],[2,2,2],[2,2,2]]
        """
        new_block_config = [(x,num_classes*y,z) for x,y,z in block_config]
        super(ClassSpecificResNet, self).__init__(new_block_config,num_classes)
    
    def make_blocks(self, block_config: List[Tuple[int]]):
        blocks = []
        in_planes = self.in_planes
        c = 0
        for b, out_planes, s in block_config:
            blocks.append(ClassSpedificResNetBlock(in_planes=in_planes, out_planes=out_planes, stride=s, num_classes=self.num_classes,name = f"{c} [block {in_planes},{in_planes}] -1"))
            for _ in range(b-1):
                blocks.append(ClassSpedificResNetBlock(in_planes = out_planes, out_planes = out_planes, stride=1,num_classes=self.num_classes,name = f"{c} [block {out_planes},{out_planes}] {_}"))
            c+=1
            in_planes = out_planes
            
        return blocks
    
    def forward(self, x,y=None):
        #X : (32,3,32,32)
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out) # (32,20,32,32)
        attentions = []
        for layer in self.layers:
            out,attention = layer(out,y)
            attentions.append(attention)
        

        out = F.avg_pool2d(out, 4)  # self.avg_pool(out)
        out = nn.Flatten()(out)
        logits = self.fc(out)
        return logits,attentions