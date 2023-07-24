import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms

def plot_losses(ax, test_losses, forget_losses, train_losses=None, name="Loss"):
    # plot losses on train and test set

    ax.set_title(f"Test, forget, and train {name} distribution")
    bins = np.histogram(np.hstack((test_losses, forget_losses, train_losses)), bins=20)[
        1]  # get the bin edges
    if train_losses is not None:
        X = (test_losses, forget_losses, train_losses)
        weights = (np.ones_like(test_losses)/len(test_losses), np.ones_like(forget_losses) /
                   len(forget_losses), np.ones_like(train_losses)/len(train_losses))
        labels = ("Test set", "Forget set", "Train set")
    else:
        X = (test_losses, forget_losses)
        weights = (np.ones_like(test_losses)/len(test_losses),
                   np.ones_like(forget_losses)/len(forget_losses))
        labels = ("Test set", "Forget set")

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
    classes = np.array(('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'))
    if len(images.shape) == 4: #this signifies batch of images
        images= inverse_normalize(images)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title(f"{title},{classes[labels.detach().cpu()]}")
        ax.set_xticks([])
        ax.set_yticks([])
        #convert images to 0 1 torch with first diamension of batch
        images = (images - images.min(dim=1, keepdim=True)[0] ) / images.max(dim=1, keepdim=True)[0]
        ax.imshow(make_grid(images.detach().cpu(), nrow=4).permute(1, 2, 0))
        plt.savefig(filename)
        plt.close()
