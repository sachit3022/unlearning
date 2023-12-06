import os
import re
import torch
from torchvision.models.resnet import resnet18
from collections import Counter
import shutil


if __name__ == "__main__":
    #count number of files in models folder
    path, dirs, files = next(os.walk("/research/hal-gaudisac/unlearning/neurips-submission/bu_unlearn_celeba"))
    
    
    cifar_files = []
    celeba_files = []
    for file in files:
        try:
            
            if torch.load(path + "/"+file,map_location="cuda:0")["epoch"]>4:
                shutil.move(path + "/"+file, "models/")
                print("moved", file)
                #print(torch.load(path + "/"+file,map_location="cuda:0")["history"]["val_accuracy"][-1])
                #cifar_files.append(file)
            elif torch.load(path + "/"+file,map_location="cuda:0")["history"]["val_accuracy"][-1] > 0.80:
                #shutil.move(path + "/"+file, "models/")
                #celeba_files.append(file)
                pass
            else:
                print( torch.load(path + "/"+file,map_location="cuda:0")["epoch"], torch.load(path + "/"+file,map_location="cuda:0")["history"]["val_accuracy"][-1], file)
        except:
            pass

    
    print("CIFAR files: ", Counter(cifar_files))
    print("CelebA files: ", Counter(celeba_files))

            
