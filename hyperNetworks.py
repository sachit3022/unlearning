from typing import Any, Callable, Dict, Optional, Tuple, Union
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.hooks import RemovableHandle
from torch.utils.data import DataLoader, Dataset
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from config import Config
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris["data"]
y = iris["target"]
names = iris["target_names"]
feature_names = iris["feature_names"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2
)

"""
class BaseModel(nn.Module):
    def __init__(self, input_dim,output_dim,activation=None):
        super(BaseModel, self).__init__()
        self.W = nn.Parameter(torch.rand(input_dim,output_dim,input_dim))
        self.activation = activation
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        P = torch.tensordot(x,self.W, dims=(1))
        out = torch.einsum('ijk,ik->ij', (P,x))
        if self.activation:
            out = self.activation(out)
        out = self.bn(out)
        return out,P

class Model(nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super(Model,self).__init__()
        hidden_dim =5
        self.base_model = nn.Sequential(*[BaseModel(input_dim,output_dim,nn.ReLU())])#+[BaseModel(hidden_dim,output_dim,activation=None)]
    def forward(self, x):
        #write something to save the weights
        all_hyper_model_weights = []        
        for layer in self.base_model:
            x,hyper_model_weights = layer(x)
            all_hyper_model_weights.append(hyper_model_weights)
        
        return x,all_hyper_model_weights

            

class LinearModel(nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super(LinearModel,self).__init__()
        hidden_dim = 5
        self.linear = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim,output_dim))
    def forward(self, x):
        return self.linear(x)




model     = Model(X_train.shape[1],3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1,weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,  eta_min=1e-6, last_epoch=-1)
loss_fn   = nn.CrossEntropyLoss()


EPOCHS  = 200
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

loss_list     = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))



for epoch in tqdm.trange(EPOCHS):

    y_pred,weights = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        y_pred,weights= model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

print(accuracy_list[-1])
print([x for x in zip(weights[0],F.softmax(y_pred),y_test)])
print(model.base_model[0].W)
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs")
fig.savefig(os.path.join(Config.LOG_PATH.value, "hyperNetworks.png") )


# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network


# Split the data set into training and testing
"""
# how will the model parameteris change if we remove one data at a time

# chose 10 index at random


datapoints = 100
dataset = []
for i in range(512):
    index = np.random.choice(X_train.shape[0], 10, replace=False)
    X_train_new = X_train[index]
    y_train_new = y_train[index]
    model = LogisticRegression()
    model.fit(X_train_new, y_train_new)
    dataset.append((X_train_new, model.coef_.flatten()))


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=4, num_heads=2, batch_first=True
        )
        self.linear1 = nn.Linear(4, 4 * 3)
        self.norm1 = nn.LayerNorm(4)
        self.dropout1 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, attn_weights = self.attention(x, x, x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)
        return x


class WeightDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


model = TransformerBlock()
EPOCHS = 100
train_set, test_set = torch.utils.data.random_split(
    WeightDataset(dataset), [0.8, 0.2])
train_loader = DataLoader(
    train_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
model.train()
model.to("cuda:1")

for epoch in tqdm.trange(EPOCHS):
    for x, y in train_loader:
        x = x.to("cuda:1").to(torch.float32)
        y = y.to("cuda:1").to(torch.float32)
        y_pred = model(x).mean(axis=1)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    total_loss = 0
    count = 0
    for x, y in test_loader:
        x = x.to("cuda:1").to(torch.float32)
        y = y.to("cuda:1").to(torch.float32)
        y_pred = model(x).sum(axis=1)
        total_loss += loss_fn(y_pred, y).item() * len(y)
        count += len(y)
    print(total_loss / count)
    print(y_pred, y)
    # suppose I use the model
    print(x @ y_pred[0].T, y_train)
