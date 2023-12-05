"""Launch to see if everything works."""
from unl_algorithms.pruning import unlearn
from unl_algorithms.pruning import load_masked_state_dict
from torchvision.models import resnet18
def main():
    pretrained_ckpt_path = "/research/hal-datastore/datasets/processed/Unlearning/CIFAR-10/pretrained_models/weights_resnet18_cifar10.pth"
    out_model = unlearn(
        pretrained_model_or_path=pretrained_ckpt_path,
        seed=0,
        device="cuda:0",
        max_epochs=1
    )

    
    checkpoint = out_model.cpu().state_dict()
    
    new_model = resnet18(weights=None, num_classes=10)
    
    new_model = load_masked_state_dict(new_model, checkpoint)


    
if __name__ == "__main__":
    main()