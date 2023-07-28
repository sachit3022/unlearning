import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import torch
import seaborn as sns

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
        plt.title(f"{title},{CIFAR_10_CLASSES[labels.detach().cpu()]}")
        ax.set_xticks([])
        ax.set_yticks([])
        #convert images to 0 1 torch with first diamension of batch
        #images = (images - images.min(dim=1, keepdim=True)[0] ) / images.max(dim=1, keepdim=True)[0]
        ax.imshow(make_grid(images.detach().cpu(), nrow=4).permute(1, 2, 0))
        plt.savefig(filename)

    plt.close()

def plot_confusion_matrix(cm,filename=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CIFAR_10_CLASSES, yticklabels=CIFAR_10_CLASSES, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


