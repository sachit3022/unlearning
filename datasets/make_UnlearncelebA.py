import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import csv

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CelebA
import torch
from torch.utils.data import Dataset
from celeba_dataset import UnlearnCelebADataset
# Helper functions for loading the hidden dataset.
def load_example(image,image_id,age_group,age,person_id):
    #age and age group are the same for now and they are indicative of the label class.
    
    result = {
        'image': image.to(torch.float32),
        'image_id': image_id,
        'age_group': age_group,
        'age': age,
        'person_id':person_id
    }
    return (image.to(torch.float32),age_group)


def _load_csv(filename, header=None):
    """
    Load a CSV file into a Pandas DataFrame.
    Copied from CelebA dataset.
    """
    with open(os.path.join("data", "celeba", filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
    if type(header) == int:
        headers = ["img_id"]+data[header][:-1]
        data = data[header + 1 :]
    else:
        headers = header
    data_int = [[row[0]]+list(map(int, row[1:])) for row in data]
    return pd.DataFrame(data_int,columns=headers)


def make_retain_forget_dataset():
    #2% of train dataset
    identity = _load_csv("identity_CelebA.txt",header=["img_id", "identity"])
    attr = _load_csv("list_attr_celeba.txt", header=1)
    split  =  _load_csv("list_eval_partition.txt",header=["img_id", "split"])
    df = pd.merge(identity,attr,on="img_id")
    df = pd.merge(df,split,on="img_id")
   
    df["class_label"] = df.apply(lambda x:int(str((x["Male"]+1)//2)+str((x["Gray_Hair"]+1)//2)+str((x["Chubby"]+1)//2),2),axis=1)
    #sample 2% of the dataset
    new_df = df[df["split"]==0].sample(frac=0.02).loc[ (df["class_label"]==0) | (df["class_label"]==4)  ]
    df.loc[df["split"]==0,"split"]= df.apply(lambda x: 4 if x["img_id"] in new_df["img_id"].values else 3,axis=1)[df["split"]==0]
    df[["img_id","split"]].to_csv("data/celeba/list_unlearn_eval_partition.txt",sep=" ",index=False,header=False)

def plot_label_distribution(dataset,prefix="train"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.hist(labels)
    plt.title("Class label distribution")
    plt.xlabel("Class label")
    plt.ylabel("Frequency")
    plt.savefig(f"{prefix}_class_label_distribution.png")
    plt.clf()
    return dict(zip(unique, counts))

def percentage_same_across_id():
    """Bald                   0.938096
    Mustache               0.865481
    Gray_Hair              0.863712
    Sideburns              0.824506
    Goatee                 0.820379
    Young                  0.817923
    Double_Chin            0.800039
    Chubby                 0.784612
    5_o_Clock_Shadow       0.734401
    No_Beard               0.718581
    Wearing_Necktie        0.718385
    Big_Lips               0.704628
    Eyeglasses             0.686745
    Rosy_Cheeks            0.684386
    Wearing_Hat            0.681340
    Blond_Hair             0.678392
    Pale_Skin              0.665029
    Receding_Hairline      0.650781
    Bushy_Eyebrows         0.598998
    Blurry                 0.587698
    Wearing_Lipstick       0.578068
    Wearing_Necklace       0.547018
    Heavy_Makeup           0.544463
    Big_Nose               0.537781
    Bangs                  0.532966
    Wearing_Earrings       0.517736
    Arched_Eyebrows        0.487275
    Black_Hair             0.464675
    Narrow_Eyes            0.427827
    Brown_Hair             0.427140
    Pointy_Nose            0.397760
    Bags_Under_Eyes        0.375356
    Wavy_Hair              0.369461
    Straight_Hair          0.366808
    Attractive             0.340375
    Oval_Face              0.335069
    High_Cheekbones        0.173529
    Smiling                0.144738
    Mouth_Slightly_Open    0.127641"""

    identity = _load_csv("identity_CelebA.txt",header=["img_id", "identity"])
    attr = _load_csv("list_attr_celeba.txt", header=1)
    df = pd.merge(identity,attr,on="img_id")
    df.drop("img_id",axis=1,inplace=True)
    new_df = df.groupby("identity").mean()
    id_share_prop = new_df.astype(int) == new_df
    return id_share_prop.sum(axis=0).sort_values(ascending=False)/len(id_share_prop)

class CelebADataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train'):
        super().__init__()
        self.transforms = transforms.Compose
        (               
            [   
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.examples= CelebA(root="data",split=split, transform=self.transforms)
        self.database_len =   len(self.examples)
        self.attr = self.examples.attr

    def __len__(self):
        return self.database_len

    def __getitem__(self, idx):
        #we will be doing constructing a 3 class classification problem by using binary encoding.
        #Male:20, Gray_Hair:17, Chubby:13
        image,labels = self.examples[idx]
        person_id = self.examples.identity[idx].item()
        class_label = int("".join(map(lambda x : str(x.item()) , (labels[[20,17,13]] + 1)//2)),2)
        example = load_example(image,idx,class_label,class_label,person_id)
        return example
def compress_images(root):
    os.chdir(root)
    for img_name in tqdm(os.listdir('img_align_celeba')):
        im = Image.open('img_align_celeba/' + img_name)
        im = im.resize((32,32), resample=Image.NEAREST)
        im.save('img_align_celeba/' + img_name)
        im.close()
    return   

if __name__ == "__main__":

    #compress_images("data/celeba")
    make_retain_forget_dataset()

    ##### Plot class label distribution #########
  
    for split in ["train","forget","retain","valid", "test"]:
        print(f"Plotting {split} class label distribution")
        dataset = UnlearnCelebADataset(split=split)
        plot_label_distribution(dataset,prefix=split)
    
    dataset = UnlearnCelebADataset(split="train")
    class_weights = plot_label_distribution(dataset,prefix="train")
    class_sample_count = torch.tensor([class_weights[i] for i in range(len(class_weights))])
    weight = 1. / class_sample_count
    print(weight)

