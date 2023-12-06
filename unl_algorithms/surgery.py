import os
import requests
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from torchvision.models import resnet18


def load_cifar():
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )

    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)

    return forget_set, retain_set


def get_grad_shape(model):
    grad_inf = [(item.numel(), item.shape) for item in model.parameters()]
    grad_numel, grad_dim = zip(*grad_inf)
    grad_numel = torch.tensor(grad_numel).cumsum(dim=0)
    return grad_numel, grad_dim


def get_grad_vector(model):
    return torch.hstack([item.grad.ravel() for item in model.parameters() if item.requires_grad])


def get_orth_grad(grad_1, grad_2):
    return -(grad_1 - (grad_1 @ grad_2) / (grad_2 @ grad_2) * grad_2)


def update_model(model, grad_vec, grad_numel, grad_dim):
    st_i = 0
    for i, item in enumerate(model.parameters()):
        item.grad = torch.reshape(grad_vec[st_i:grad_numel[i]], grad_dim[i])
        st_i = grad_numel[i]


def get_l1_loss(model, lmb=1e-2):
    loss = 0
    for item in model.parameters():
        loss += F.l1_loss(item, torch.zeros_like(item))

    return lmb * loss


def unlearning(net, seed, device,datasets):
    torch.manual_seed(seed)
    retain_set, forget_set = datasets["retain"], datasets["forget"]

    forget_loader = DataLoader(
        forget_set, batch_size=2048, shuffle=True, num_workers=8, persistent_workers=True
    )
    retain_loader = DataLoader(
        retain_set, batch_size=1024, shuffle=True, num_workers=8,persistent_workers=True
    )
    epochs = 4
    net.to(device)
    net.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=9e-2, momentum=0.95, weight_decay=0)
    grad_numel, grad_dim = get_grad_shape(net)
    n_retain = len(retain_set)

    for _ in range(3):
        for inputs, targets in tqdm(forget_loader):
            retain_index = np.random.choice(n_retain, len(targets))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            grad_f = get_grad_vector(net)

            optimizer.zero_grad()
            inputs, targets = zip(*[retain_set[i] for i in retain_index])
            inputs = torch.torch.stack(inputs)
            targets = torch.tensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            grad_r = get_grad_vector(net)
            orth_grad = get_orth_grad(grad_f, grad_r)
            update_model(net, orth_grad, grad_numel, grad_dim)
            optimizer.step()

    optimizer = optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    for _ in range(epochs):
        for inputs, targets in tqdm(retain_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss += get_l1_loss(net, lmb=1e-3)
            loss.backward()
            optimizer.step()
        scheduler.step()
    

    return net


if __name__=='__main__':
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)

    device = torch.device("cuda:4")
    weights_pretrained = torch.load(local_path, map_location=device)
    ft_model = resnet18(weights=None, num_classes=10)
    ft_model.load_state_dict(weights_pretrained)
    ft_model.to(device)
    ft_model = unlearning(ft_model, 0, device)
