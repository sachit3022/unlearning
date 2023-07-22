import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


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


def plot_image_grid(images, labels=None, filename=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(f"Sample images from CIFAR10 dataset {labels}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
    plt.savefig(filename)
