import os

import torch
import torch.nn as nn
import torch.optim as optim 
from config import Config

from typing import List
from torch.utils.data import DataLoader

DEVICE = Config.DEVICE.value

class MiaBaseModel(nn.Module):
  def __init__(self,in_features,out_features) -> None:
      super().__init__()
      self.linear = nn.Linear(in_features=in_features,out_features=out_features)
      self.bn = nn.BatchNorm1d(out_features)
      self.relu = nn.ReLU()

  def forward(self,x):
      out = self.linear(x)
      out = self.bn(out)
      out = self.relu(out)
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
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
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
      train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True,num_workers=2)
      for _ in range(self.epocs):
        for inputs,targets in train_dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
      return self
    
    def predict(self,X):
        return torch.argmax(self(X),dim=1)
        
    def get_params(self,deep=True):
        #this is a hack to make cross validation work with torch model change it once you find time.

        model_path = os.path.join(Config.MODEL_DIR.value,"mia.pt")
        torch.save(self.state_dict(),model_path)
        return  {"model_config":self.model_config,"num_classes":self.num_classes,"pretrained_model":model_path}
    def set_params(self, **params):
        self.load_state_dict(torch.load(params["pretrained_model"]))