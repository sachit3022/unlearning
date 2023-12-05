"""Launch to see if everything works."""
from unl_algorithms.pruning import unlearn

def main():
    pretrained_ckpt_path = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10/pretrained_models/weights_resnet18_cifar10.pth"
    unlearn(
        pretrained_model_or_path=pretrained_ckpt_path,
        seed=0,
        device="cuda:0"
    )
    
if __name__ == "__main__":
    main()