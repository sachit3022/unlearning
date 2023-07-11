import matplotlib.pyplot as plt
import numpy as np


def plot_losses(ax,test_losses,forget_losses):
    # plot losses on train and test set

    ax.set_title(f"Pre-trained model.\nAttack accuracy:")
    ax.hist(test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    ax.hist(forget_losses, density=True, alpha=0.5, bins=50, label="Forget set")
    ax.set_xlabel("Loss", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xlim((0, np.max(test_losses)))
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=14)
    return 


