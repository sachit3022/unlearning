from torch import nn, optim
import torch


def unlearning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader,DEVICE):
    """Simple unlearning by finetuning."""
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()

    for ep in range(epochs):
        net.train()
        for sample in retain_loader:
            inputs = sample["image"]
            targets = sample["age_group"]
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()     
    net.eval()
    return net