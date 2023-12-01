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
from torchvision.models import resnet18
from torch.utils.data import DataLoader,Subset
from celeba_dataset import UnlearnCelebADataset
from evaluation.SVC_MIA import collect_prob,entropy
import seaborn as sns
import pickle



CIFAR_10_CLASSES = np.array(('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'))

def plot_losses(ax, test_losses, forget_losses, train_losses=None, name="Loss"):
    # plot losses on train and test set

    ax.set_title(f"Retain, Test, and Forget {name} distribution")

    if train_losses is not None:
        X = (test_losses, forget_losses, train_losses)
        weights = (np.ones_like(test_losses)/len(test_losses), np.ones_like(forget_losses) /
                   len(forget_losses), np.ones_like(train_losses)/len(train_losses))
        labels = ("Test set", "Forget set", "Retain set")
    else:
        X = (test_losses, forget_losses)
        weights = (np.ones_like(test_losses)/len(test_losses),
                   np.ones_like(forget_losses)/len(forget_losses))
        labels = ("Test set", "Forget set")
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



import matplotlib as mpl
from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
from scipy.stats import multivariate_normal

def train(model,X,y,optim,epochs=1000,ascent=False):
    model.train()
    losses = []
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        optim.zero_grad()
        y_pred = model(torch.from_numpy(X).float())
        loss = loss_fn(y_pred.squeeze(),torch.from_numpy(y).float())
        if ascent:
            loss = -loss
        losses.append(loss.item())
        loss.backward()
        optim.step()
    return losses

class LossSurface(object):
    def __init__(self, model, dataloader,device):
        self.model_ = deepcopy(model).to(device)
        self.model_.eval()
        self.inputs_ = dataloader
        self.coords = Coordinates(self.model_)            
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device



    def compile(self, range, points):
        a_grid = torch.linspace(-1.0, 1.0, points) ** 3 * range
        b_grid = torch.linspace(-1.0, 1.0, points) ** 3 * range
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                self.model_.load_state_dict(self.coords(a, b))

                #print(self.model_.state_dict(),self.coords(a,b))
                total_loss = 0.0
                count = 0
                for batch in self.inputs_:
                    loss = self.loss_fn(self.model_(batch[0].to(self.device)), batch[1].to(self.device))
                    total_loss += loss.item()
                    count += 1
                    if count > 20:
                        break
                loss = total_loss / count
                loss_grid[j, i] = loss
        self.model_.load_state_dict(self.coords.origin_())
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid
        

    def plot(self, range=1.0, points=24, levels=20, ax=None, **kwargs):
        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        if ax is None:
            _, ax = plt.subplots()

            ax.set_title("The Loss Surface")
            ax.set_aspect("equal")
        # Set Levels
        min_loss = zs.min()
        max_loss = zs.max()
        levels = torch.exp(
            torch.linspace(
                torch.math.log(min_loss+1e-20), torch.math.log(max_loss+1e-20), levels
            )
        )
        CS = ax.contour(
            xs,
            ys,
            zs,
            levels=levels,
            linewidths=0.75,
            norm=mpl.colors.LogNorm(vmin=min_loss+1e-20, vmax=max_loss * 2.0),
            cmap="magma",
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        return ax
    
    def plot_single_point(self, ax,new_model,**kwargs):
        x = self.coords.coordinate_shifting(new_model)
        ax.scatter(x[0].detach().cpu().item(), x[1].detach().cpu().item(), **kwargs)
        return ax

class Coordinates:
    def __init__(self,model,input=None,output=None) -> None:
        #store the shape of the model
        self.model = deepcopy(model)
        self.origin = deepcopy(torch.cat([x.data.flatten() for x in self.model.parameters()]))
        self.x1 = torch.rand_like(self.origin)
        self.x2 = torch.rand_like(self.origin)
        self.x1 = self.x1 / torch.norm(self.x1) 
        self.x2 = self.x2 / torch.norm(self.x2)
        self.x1 = self.x1 - (self.x1 @ self.x2) * self.x2
        self.x1 = self.x1 / torch.norm(self.x1)
        """
        self.x1 = torch.zeros_like(self.origin)
        self.x1[0] =1
        self.x2 = torch.zeros_like(self.origin)
        self.x2[1] =1

        #self.origin = torch.zeros_like(self.origin)
        """
        if input is not None and output is not None:
            #self.pca_directions(input,output)
            pass
        
        
    def convert_to_model_weight(self,w0):
        m = deepcopy(self.model)
        start= 0
        for params in m.parameters():
            params.data = w0[start:start+math.prod(params.shape)].reshape(params.shape)
            start += math.prod(params.shape)
        return m.state_dict()

    def __call__(self,a,b):
        #reshape into original shape
        w0 = self.origin + a*self.x1 + b*self.x2
        return self.convert_to_model_weight(w0)
    
    def origin_(self):
        return self.convert_to_model_weight(self.origin)
    
    def coordinate_shifting(self,model):
        x = torch.cat([x.data.flatten() for x in model.parameters()])
        x = x - self.origin
        #project x onto x1 and x2
        a = x @ self.x1
        b = x @ self.x2
        return a,b
    def pca_directions(self,inputs,outputs):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        weights= []
        for e in range(1000):
            train(self.model, inputs, outputs,optim ,1)
            weights.append(torch.cat([x.data.flatten() for x in self.model.parameters()]))

        weights = torch.stack(weights)
        weights = weights - weights.mean(0)
        weights = weights / weights.std(0)
        weights = weights.T
        weights = torch.svd(weights).U
        weights = weights.T
        self.x1 = weights[0]
        self.x2 = weights[1]
        return 




if __name__ == "__main__":
    # plot losses on train and test set
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=8).to(device)
    #model.load_state_dict(torch.load("models/model_scratch_42_resnet18.pt",map_location=device)["model_state_dict"])
    model.load_state_dict(torch.load("models/model_unl.pt",map_location=device)["model_state_dict"])
    retain_dataset = UnlearnCelebADataset("retain",100)
    retain_loader = DataLoader(retain_dataset,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    loss_surface = LossSurface(model,retain_loader,device)
    loss_surface.compile(10, 24)
    ax = loss_surface.plot()
    #save ax
    ax.figure.savefig("loss_surface.png")
    ax.figure.clf()
    pickle.dump(loss_surface,open("loss_surface.pkl","wb"),protocol=pickle.HIGHEST_PROTOCOL)


    """
    train_dataset = UnlearnCelebADataset("train")
    retain_dataset = UnlearnCelebADataset("retain")
    forget_dataset = UnlearnCelebADataset("forget")
    valid_dataset = UnlearnCelebADataset("valid")
    test_dataset = UnlearnCelebADataset("test")

    retain_loader_train = DataLoader(Subset(train_dataset,range(0,len(test_dataset))),batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    forget_loader = DataLoader(forget_dataset,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    
    shadow_train_prob, shadow_train_labels = collect_prob(retain_loader_train, model,device)
    shadow_test_prob, shadow_test_labels = collect_prob(test_loader, model,device)
    target_test_prob, target_test_labels = collect_prob(forget_loader, model,device)

    shadow_train_conf = entropy(shadow_train_prob).flatten().detach().cpu().numpy()
    shadow_test_conf = entropy(shadow_test_prob).flatten().detach().cpu().numpy()
    target_test_conf = entropy(target_test_prob).flatten().detach().cpu().numpy()

    #shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None]).flatten().detach().cpu().numpy()
    #shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None]).flatten().detach().cpu().numpy()
    #target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None]).flatten().detach().cpu().numpy()

    fig,ax = plt.subplots()
    plot_losses(ax, shadow_test_conf, target_test_conf, shadow_train_conf, name="Entropy")
    fig.savefig("Entropy.png")
    fig.clf()
    """