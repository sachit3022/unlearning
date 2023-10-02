import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import torch
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from copy import deepcopy
import math
import cv2
irange = range

CIFAR_10_CLASSES = np.array(('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'))

def plot_losses(ax, test_losses, forget_losses, train_losses=None, name="Loss"):
    # plot losses on train and test set

    ax.set_title(f"Train, Val, and Test {name} distribution")

    if train_losses is not None:
        X = (test_losses, forget_losses, train_losses)
        weights = (np.ones_like(test_losses)/len(test_losses), np.ones_like(forget_losses) /
                   len(forget_losses), np.ones_like(train_losses)/len(train_losses))
        labels = ("Train set", "Val set", "Test set")
    else:
        X = (test_losses, forget_losses)
        weights = (np.ones_like(test_losses)/len(test_losses),
                   np.ones_like(forget_losses)/len(forget_losses))
        labels = ("Train set", "Val set")
    bins = np.histogram(np.hstack(X), bins=20)[
        1]  # get the bin edges

    ax.hist(X, density=False, alpha=0.5, bins=bins,
            weights=weights, label=labels)

    ax.set_ylabel("Percentage Samples", fontsize=14)
    ax.legend(frameon=False, fontsize=14)
    return

inverse_normalize = transforms.Normalize(
   mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
   std=[1/0.2023, 1/0.1994, 1/0.2010]
)


def plot_image_grid(images, labels=None, filename=None,title=None):
    #cifar 10 classes

    if len(images.shape) == 4: #this signifies batch of images
        images= inverse_normalize(images)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title(f"{title},{labels.detach().cpu()}")
        ax.set_xticks([])
        ax.set_yticks([])
        #convert images to 0 1 torch with first diamension of batch
        #images = (images - images.min(dim=1, keepdim=True)[0] ) / images.max(dim=1, keepdim=True)[0]
        ax.imshow(make_grid(images.detach().cpu(), nrow=4).permute(1, 2, 0))
        plt.savefig(filename)

    plt.close()

def plot_confusion_matrix(cm,filename=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()





class GradCamWrapper:
    def __init__(self,model,device) -> None:
        self.model = deepcopy(model)
        target_layers = [self.model.layer4[-1]]
        self.cam =  GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)
        self.new_d_model = deepcopy(model)
        self.device = device
        
    def __call__(self,batch_data):

        images,labels = batch_data[0].clone().to(self.device), batch_data[1].clone().to(self.device)
        visualisation = torch.stack([self.apply_grad_cam_on_image(x_i) for x_i in torch.unbind(images, dim=0)], dim=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(make_grid(visualisation.cpu().permute(0,3,1,2),nrow=4).permute(1, 2, 0))
        return fig,ax
    
    def apply_grad_cam_on_image(self,img):
        
        mask = self.cam(input_tensor=img.unsqueeze(0),targets = [ClassifierOutputTarget(1)])
        
        
        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img -= img.min()
        img /= img.max()
        img = img.astype(np.float32)

        return torch.tensor(show_cam_on_image(img =img, mask=mask[0], use_rgb=True))
    


def plot_simple_image_grid(images,filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(make_grid(images.cpu()[:8],nrow=4,padding = 5,pad_value=1).permute(1, 2, 0))
    plt.savefig(filename)
    plt.close()
