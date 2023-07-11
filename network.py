import os
import requests

import torch
import torch.nn as nn
import torch.optim as optim 
from config import Config

import random
from typing import List,Optional,Tuple
from torch.utils.data import DataLoader

DEVICE = Config.DEVICE.value

# to track all the losses and other parameters. start with loss and accuracy.
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MiaBaseModel(nn.Module):
  def __init__(self,in_features,out_features) -> None:
      super().__init__()
      self.linear = nn.Linear(in_features=in_features,out_features=out_features)
      self.bn = nn.BatchNorm1d(out_features)
      self.dropout = nn.Dropout(0.2)
      self.relu = nn.ReLU()

  def forward(self,x):
      out = self.linear(x)
      out = self.bn(out)
      out = self.relu(out)
      out = self.dropout(out)
      return out

class MiaModel(nn.Module):
    def __init__(self,model_config:List[int],num_classes:int,pretrained_model=None) -> None:
        super().__init__()
        self.model_config= model_config
        self.num_classes = num_classes

        self.layers = nn.Sequential(*self.make_layers(model_config+[num_classes]))
        self.softmax = nn.Softmax(dim=1)
        self.epocs = 500
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epocs)
        if pretrained_model is not None:
            self.load_state_dict(torch.load(pretrained_model))

    def make_layers(self,model_config):
        layers = []
        in_channel = model_config[0]
        for out_channel in model_config[1:]:
            layers.append(MiaBaseModel(in_channel,out_channel))
            in_channel = out_channel
        return layers
            
    def forward(self,x):
      out = x
      for l in self.layers:
          out = l(out)
        
      out =  self.softmax(out)
      return out
    
    def fit(self,train_dataset,y=None):
        self.to(DEVICE)
        self.train()
        for _ in range(self.epocs):
            inputs,targets = train_dataset[:]
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return self
    
    def predict(self,X,y=None):
        return torch.argmax(self(X),dim=1)
    
    def score(self,X,y=None):
        self.eval()
        inputs,targets = X[:]
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = self(inputs)
        return self.accuracy(outputs,targets)
    
    def accuracy(self,outputs,targets):
        _,preds = torch.max(outputs,dim=1)
        return torch.sum(preds == targets).item() / len(preds)
        
    def get_params(self,deep=True):
        #this is a hack to make cross validation work with torch model change it once you find time.
        model_path = f"models/mia.pt" #takes only strings thats strange

        torch.save(self.state_dict(),model_path)
        #bug in sklearn so dowmgrde to 0.21.2
        return  {"model_config":self.model_config,"num_classes":self.num_classes,"pretrained_model":model_path}
    
    def set_params(self, **params):
        self.load_state_dict(torch.load(params["pretrained_model"]))


#resnet implementation so we cam control the capacity of the model.

class BasicBlock(nn.Module):
    def __init__(self,in_planes:int,out_planes:int,kernel_size:int=3,stride:Optional[int]=None,groups:int=1,padding:Optional[str]="same"):
        super().__init__()
        # Conv3D
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,groups=groups,padding=1,dilation=1,stride=stride)
        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=kernel_size,groups=groups,padding=1,dilation=1,stride=1)
        #what happens if inplanes and outplanes are not the same. how do we perform sum.
        #What happens if HXW changes. 
        #authors propose the idea of zero padding, to the input and not the output. planes will grow but H and W will shrink
        self.downsample =None
        if stride!=1 or in_planes !=out_planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(out_planes))
        
        # what if we use LayerNorm instead of BatchNorm.
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        #activation of relu
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
      
        #to match diamensions of x with that of output.
        if self.downsample:
            x = self.downsample(x)
        
        out += x
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self,block_config:List[Tuple[int]],num_classes:int=10):
        super().__init__()
        self.block_config = block_config
        
        self.in_planes = block_config[0][1]
        self.out_layer = block_config[-1][1]

        self.conv1 = nn.Conv2d(3,self.in_planes,kernel_size=3,stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    

        self.layers = nn.Sequential(*self.make_blocks(block_config))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.out_layer,num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self,x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.maxpool(out)

        for layer in self.layers:
            out = layer(out)
            
        out = self.avg_pool(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out
    def predict_pobs(self,X):
        return self.softmax(self(X))
    
    def make_blocks(self,block_config:List[Tuple[int]]):
        blocks= []
        in_planes = self.in_planes
        for b,out_planes in block_config:
            blocks.append(BasicBlock(in_planes,out_planes,stride=2))
            for _ in range(b) :
                blocks.append(BasicBlock(out_planes,out_planes,stride=1))
            in_planes = out_planes
        return blocks
    
    @property
    def name(self):
        return f"ResNet_{'-'.join( [ 'x'.join(tuple(map(str, tup))) for tup in self.block_config])}"


if os.path.exists(os.path.join(Config.MODEL_DIR.value,"weights_resnet18_cifar10.pth")):
    # download pre-trained weights
    response = requests.get(
        "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
    )
    open(os.path.join(Config.MODEL_DIR.value,"weights_resnet18_cifar10.pth"), "wb").write(response.content)
