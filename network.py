from typing import List, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

#just get resnet18 features
class ResNet18(nn.Module):
    def __init__(self, pretrained: bool=True) -> None:
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.fc = nn.Identity()

        #muti task model
        self.model.fc1 = nn.Linear(512, 2)
        self.model.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        return self.model(x)


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
        #import pdb;pdb.set_trace()
        out = x
        for l in self.layers:
            out = l(out)

        #out =  self.sigmoid(out)
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
        self.relu = nn.ReLU(inplace=True)
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

        out += x
        out = self.relu(out)

        return out

    def predict_pobs(self, X):
        out = self.forward(X)
        return self.softmax(out)


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
        return f"ResNet_{'-'.join( [ 'x'.join(tuple(map(str, tup))) for tup in self.block_config])}"

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)
