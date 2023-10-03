import copy
import torch
import logging
import config
import network
import trainer as tr
from trainer import Trainer, TrainerSettings, count_parameters
from dataset import create_cifar10_dataloaders


from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List
from torch import einsum
from einops import rearrange, repeat
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
import argparse
from matplotlib import pyplot as plt
from plots import plot_simple_image_grid

def main(args):
    

    # set loggers
    logger = logging.getLogger()

    # load model
    net = getattr(network, args.model.name)(**args.model.model_args)
    net.to(args.device)
    print(net)
    print(f"Model has {count_parameters(net)} parameters")    
    train_dataloaders = create_cifar10_dataloaders(config = args)

    optimizer  = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.00001,betas = [0.9,0.999])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    for i in trange(100):
        total_acc = 0
        count = 0
    
        net.train()
        for batch_data in train_dataloaders.train:
        
            images,labels = batch_data[0].to(args.device), batch_data[1].to(args.device)
            #if random.random() < args.mixup and i<50:
            #    outputs,attentions = net(images,labels)
            #else:
            outputs,attentions = net(images)
            #enforce constraints on non_label ones should be zero
            loss = criterion(outputs,labels) + 0.1*torch.mean(torch.abs(torch.sum(attentions,dim=1)-1))


            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            total_acc += acc
            count = count + 1

        print(f"Epoch train {i} loss {loss} acc {total_acc/count}")
            
        total_acc = 0
        count = 0
        net.eval()
        for batch_data in train_dataloaders.test:
            with torch.no_grad():
                images,labels = batch_data[0].to(args.device), batch_data[1].to(args.device)
                outputs,attentions = net(images)
                loss = criterion(outputs,labels)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                total_acc += acc
                count = count + 1
        print(f"Epoch test {i} loss {loss} acc {total_acc/count}")
        torch.save(net.state_dict(), f"models/{args.experiment}_csa.pth")
    return 

def test_model(args):
    
    #load model
    net = getattr(network, args.model.name)(**args.model.model_args)
    net.to(args.device)
    #print(net)
    #print(f"Model has {count_parameters(net)} parameters")   
    net.load_state_dict(torch.load("models/csa_csa.pth",map_location=args.device))
    train_dataloaders = create_cifar10_dataloaders(config = args)

    criterion = nn.CrossEntropyLoss()
    total_acc = 0
    count = 0
    net.eval()
    for batch_data in train_dataloaders.test:
        with torch.no_grad():
            images,labels = batch_data[0].to(args.device), batch_data[1].to(args.device)
            plot_simple_image_grid(images.detach().cpu(),f"logs/attention/image.png") 
            outputs,attentions = net(images)
            loss = criterion(outputs,labels)
            print(labels,outputs.argmax(dim=1))
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            total_acc += acc
            count = count + 1 
        break

    print(f"Test loss {loss} acc {total_acc/count}")
    return
    






if __name__ == "__main__":
    """
    python ClassAttention.py -cf "class_attn.yaml" -d "cuda:2" -exp "CSA"

    """ 
    parser = argparse.ArgumentParser(description="Class Attention")
    parser.add_argument("-m", "--mixup", type=float, default=0, help="Mix of path vs attention")
    parser.add_argument("-cf", "--config_file", default="class_attn.yaml", type=str, help="yaml config file, default: class_attn.yaml")
    args = config.set_config(parser)
    test_model(args)
    #main(args)



    #test_model(args)

    # Comments:
    # not much progress while training the model on the forget set.
    # why is there no change in gradients when there is change in the loss? investigate the mia_dataset and comeup with solution.
    # no matter what the train and test are the same
    # by using label smoothing the mia score decreases and test and train remians the same.
    # best to test on faces dataset. vggface2
    # think of ideas to debug the gradient problem.


