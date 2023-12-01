
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import trange
import argparse
from matplotlib import pyplot as plt
from plots import plot_simple_image_grid
from einops import rearrange, repeat


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.selfsplit = nn.Linear(10,10*10)
        self.qkv = nn.Linear(10,10*3)
    def forward(self, x):
        x = self.selfsplit(x)
        x = rearrange("b (w h) -> b w h",x,w=3)
        q,k,v = self.qkv(x).chunk(3,dim=-1)
        att = F.softmax(torch.matmul(q,k),dim=-1)
        x = torch.matmul(att,v)
        x = x.view(-1,10)
        return x
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.attention = AttentionNet()
        self.fc2 = nn.Linear(10, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x







def main():
    #load mnist data
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="data", train=True,
            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
            shuffle=False, num_workers=2)
    testset = torchvision.datasets.MNIST(root='data', train=False,
            download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
            shuffle=False, num_workers=2)
    
    #simple MLP

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    #train  
    for epoch in trange(10):  # loop over the dataset multiple times
        total_acc = 0
        count = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            total_acc += acc
            count = count + 1
        print(f"Epoch train {epoch} loss {loss} acc {total_acc/count}")
        total_acc = 0
        count = 0
        net.eval()
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels = data
                outputs = net(inputs)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                total_acc += acc
                count = count + 1
        print(f"Epoch test {epoch} acc {total_acc/count}")
    print('Finished Training')

if __name__ == "__main__":
    main()


