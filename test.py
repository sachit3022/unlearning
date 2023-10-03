import argparse
from typing import Any
import network
from trainer import count_parameters
from config import dotformat, dotdict,seed_everything
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import UnlearnCelebA, SampleCelebA
from plots import plot_image_grid, GradCamWrapper
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random


def test(model,checkpoint,args):

    #load network
    net = getattr(network, model.name)(**model.model_args)
    net.load_state_dict(torch.load(checkpoint, map_location=args.device)["model_state_dict"])
    net.name = model.name
    net = net.to(args.device)

    print(f"Model has {count_parameters(net)} parameters")

    #load data
    #sample only posion, 

    split = "train"
    aug = "only_patch"
    dist = "p"

    net.eval()

    test_set = UnlearnCelebA(root = args.DATA_PATH, split=split,aug =aug,poison_prob=args.ps_pb,classify_across=args.cs_acc)
    test_dataloader = DataLoader(test_set, batch_size=args.BATCH_SIZE,num_workers=6,persistent_workers=True, pin_memory=True,sampler=SampleCelebA(test_set))
    test_model(net,test_dataloader,args.device,f"{split}_{dist}_{aug}")
    return net

def test_model(model,dataloader,device,exp_name = "test"):
    
    #test phase
    correct,masked,unmasked,masked_count,unmasked_count,total = 0,0,0,0,0,0
    totsl_lb = 0
    gradCam = GradCamWrapper(model,device)
    for batch_id,batch_data in tqdm(enumerate(dataloader)):
        images,labels = batch_data[0].to(device), batch_data[1].to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            masked += (predicted[batch_data[2]] == labels[batch_data[2]]).sum()
            unmasked += (predicted[~batch_data[2]] == labels[~batch_data[2]]).sum()
            totsl_lb +=labels.sum()
            masked_count += batch_data[2].sum()
            unmasked_count += (~batch_data[2]).sum()

        if random.random() < 0.01:
            for id,filter in enumerate([batch_data[2],~batch_data[2]]):
                if filter.sum() == 0:
                    continue
                exp_fig,exp_ax = gradCam([batch_data[0][filter] for x in batch_data])
                print(f"{id}_{exp_name}_{batch_id}")

                exp_ax.set_title(f"(Mask,predicted,labels) : {[(i.item(),j.item(),l.item()) for i,j,l in zip(batch_data[2][filter],predicted[filter],labels[filter])]}",loc='center', wrap=True)
                exp_fig.savefig(f"logs/explainable/{id}_{exp_name}_{batch_id}.png")
                plt.close()

    print(f"Accuracy of the network on the {total} test images: {100 * correct / total} %%"  )
    print(f"Accuracy of the network on the {masked_count} test images: {100 * masked / masked_count} %%"  )
    print(f"Accuracy of the network on the {unmasked_count} test images: {100 * unmasked / unmasked_count} %%"  )
    print(totsl_lb)
    return 

if __name__ == "__main__":
    """
    How to test your model
    python test.py -d "cuda:1" -m '{"name":"resnet50","model_args":{"num_classes":2}}' -cpt "/research/hal-gaudisac/unlearning/models/model_hair_50_0_resnet50.pt" -dp "data" -dt '{"num_workers":6, "rf_split":[0.1,0.9],"num_classes":2}'    
    """

    parser = argparse.ArgumentParser()
    
    #network hyperparameters
    parser.add_argument('-cpt',"--checkpoint",type=str,required=True)
    parser.add_argument('-m','--model', required=True, action=dotformat,
                        help='specify the model you want to test')
    parser.add_argument('-d','--device', default='cuda:1', type=str,help='device in cuda:d format to run the model on') # becase we want to adapt to mps

    #data hyperparameters
    parser.add_argument('-dp','--DATA_PATH', default="/research/hal-gaudisac/data/celeba", type=str,help='path to the dataset')
    parser.add_argument('-bs','--BATCH_SIZE', default=32, type=int,help='batch size for training')
    parser.add_argument('-dt','--data',required=True, action=dotformat, help='dict of num_classes, num_workers, rf_split')
    parser.add_argument('--cs_acc', default=2, type=int,help="2 for attractiveness and 8 for hair color")
    parser.add_argument('--ps_pb',default=0.0,type=float,help="percent of samples where Identity patch is trained.")
    
    seed_everything(42)
    args =  dotdict(vars(parser.parse_args()))
    test(args.model,args.checkpoint,args)