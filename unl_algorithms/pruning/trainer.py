import torch
from torch import nn
from .utils import enable_pruning, reset_pruning
import math


###########################
# Eval
###########################
@torch.no_grad()
def eval_step(batch, model, criterion, device):
    image,label = batch[0].to(device), batch[1].to(device)
    out = model(image)
    loss = criterion(out, label)
    
    pred = out.argmax(1)
    correct = pred.eq(label).sum()
    
    return correct.item(), loss.item()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, losses = 0, 0

    for batch in loader:
        corr, loss = eval_step(batch, model, criterion, device)
        correct += corr
        losses += loss

    acc = correct / len(loader.dataset)
    losses = losses / len(loader.dataset)
    return acc, losses

###########################
# Train
###########################
@torch.no_grad()
def prune_step(batch, model, device): # forget only
    with enable_pruning(model):
        image, label = batch[0].to(device), batch[1].to(device)
        model(image)
        
def train_step(batch, model, optimizer, criterion, device): # retain only
    image, label = batch[0].to(device), batch[1].to(device)

    optimizer.zero_grad()
    out = model(image)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()

    pred = out.detach().argmax(1)
    correct = pred.eq(label).sum()
    
    return loss.detach().item(), correct.item()


def train_with_pruning(
    prune_epochs: int,
    total_epochs: int,
    dataloaders,
    model,
    optimizer,
    criterion,
    scheduler,
    scheduler_type: str,
    device,
    reset_mask_after_pruning: bool = False
) -> None:
    def prune_epoch(epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        R, F = len(dataloaders["retain"]), len(dataloaders["forget"])
        N = R + F
        F_freq = math.ceil(N / F)
        
        # manual iteration
        retain_iter, forget_iter = iter(dataloaders["retain"]), iter(dataloaders["forget"])
        
        i, j = 0, 0
        for n in range(N):
            if n % F_freq and j < F:
                batch = next(forget_iter)
                is_forget_step = True
                j += 1
            else:
                batch = next(retain_iter)
                is_forget_step = False
                i += 1
                
            if is_forget_step:
                # compute prune mask on forget, then disable mask computation
                prune_step(batch, model, device)
            else:
                # finetune step on retain
                loss, correct = train_step(batch, model, optimizer, criterion, device)
                running_loss += loss
                running_correct += correct
                
                if scheduler_type == "on_step":
                    scheduler.step()
        
        if scheduler_type == "on_epoch":
            scheduler.step()
        retain_acc = running_correct / len(dataloaders["retain"].dataset)
        retain_loss = running_loss / len(dataloaders["retain"])
          
        test_acc, test_loss = evaluate(model, dataloaders["test"], criterion, device)
        forget_acc, forget_loss = evaluate(model, dataloaders["forget"],criterion, device)
        
        print(f"Epoch: {epoch} | retain_loss: {retain_loss:6.4f} | retain_acc: {retain_acc:6.4f} | forget_loss: {forget_loss:6.4f} | forget_acc: {forget_acc:6.4f} | test_loss: {test_loss:6.4f} | test_acc: {test_acc:6.4f} | lr: {optimizer.param_groups[0]['lr']:6.4e}")
    
    def finetune_epoch(epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        for batch_retain in dataloaders["retain"]:
            loss, correct = train_step(batch_retain, model, optimizer, criterion, device)
            running_loss += loss
            running_correct += correct
        
        scheduler.step()
          
        retain_acc = running_correct / len(dataloaders["retain"].dataset)
        retain_loss = running_loss / len(dataloaders["retain"])
          
        test_acc, test_loss = evaluate(model,dataloaders["test"], criterion, device)
        forget_acc, forget_loss = evaluate(model,dataloaders["forget"], criterion, device)
        print(f"Epoch: {epoch} | retain_loss: {retain_loss:6.4f} | retain_acc: {retain_acc:6.4f} | forget_loss: {forget_loss:6.4f} | forget_acc: {forget_acc:6.4f} | test_loss: {test_loss:6.4f} | test_acc: {test_acc:6.4f} | lr: {optimizer.param_groups[0]['lr']:6.4e}")
    
    n_p = 0
    for epoch in range(total_epochs):
        if epoch % 2 == 0 and n_p < prune_epochs:
            prune_epoch(epoch)
            n_p += 1
        else:
            finetune_epoch(epoch)
            
            if reset_mask_after_pruning:
                reset_pruning(model) # clear out masks