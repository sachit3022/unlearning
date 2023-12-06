import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from datasets import get_celeba_dataloaders,get_cifar10_dataloaders
from torchvision.models import resnet18
import os
import random
import dotmap



def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p < 1, reverse_prob.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model,device):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
            except:
                data, target = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    n_shadow_test = 1000
    n_iters = n_shadow_train // n_shadow_test

    

    accs = []
    #randomly sample from shadow_train
    for i in range(n_iters):
        X_shadow = (
            torch.cat([shadow_train[n_shadow_test*i:n_shadow_test*(i+1)],shadow_train[torch.randperm(n_shadow_test)]]).cpu()
            .numpy()
            .reshape(n_shadow_test + n_shadow_test, -1)
        )
        Y_shadow = np.concatenate([np.ones(n_shadow_test), np.zeros(n_shadow_test)])

        clf = SVC()#C=3, gamma="auto", kernel="rbf"
        clf.fit(X_shadow, Y_shadow)

        if n_target_train > 0:
            X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
            acc_train = clf.predict(X_target_train).mean()
            accs.append(acc_train)

        if n_target_test > 0:
            X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
            acc_test = 1 - clf.predict(X_target_test).mean()
            accs.append(acc_test)

    return np.mean(accs)


def compute_metrics(shadow_train_tuple, shadow_test_tuple, target_test_tuple, target_train_tuple):
    shadow_train_prob, shadow_train_labels = shadow_train_tuple
    shadow_test_prob, shadow_test_labels = shadow_test_tuple
    target_train_prob, target_train_labels = target_train_tuple
    target_test_prob, target_test_labels = target_test_tuple


    shadow_train_corr = (
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
        torch.argmax(target_train_prob, axis=1) == target_train_labels
    ).int()
    target_test_corr = (
        torch.argmax(target_test_prob, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)

    target_train_entr = entropy(target_train_prob)
    target_test_entr = entropy(target_test_prob)

   


    """
    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)

    if target_train_entr.shape[0] > 0:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test_entr.shape[0] > 0:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr
    """

    acc_corr = SVC_fit_predict(
        shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    acc_entr = SVC_fit_predict(
        shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    )
    """
    acc_m_entr = SVC_fit_predict(
        shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    )
    """
    acc_prob = SVC_fit_predict(
        shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    )
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr, #"m_entropy": acc_m_entr,
        
        "prob": acc_prob,
        "RA":shadow_train_corr.float().mean(),
        "UA":target_test_corr.float().mean(),
        "TA":shadow_test_corr.float().mean()
    }
    print(m)
    return m


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model,device):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model,device)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model,device)
    target_train_prob, target_train_labels = collect_prob(target_train, model,device)
    target_test_prob, target_test_labels = collect_prob(target_test, model,device)
    return compute_metrics((shadow_train_prob, shadow_train_labels), (shadow_test_prob, shadow_test_labels), (target_test_prob, target_test_labels), (target_train_prob, target_train_labels))


def SVC_retrain_unl(retrain_loader, forget_loader,retain_model,unl_model,device):

    shadow_train_prob, shadow_train_labels = collect_prob(retrain_loader, unl_model,device)
    shadow_test_prob, shadow_test_labels = collect_prob(forget_loader, unl_model,device)

    target_train_prob, target_train_labels = collect_prob(retrain_loader, retain_model,device)
    target_test_prob, target_test_labels = collect_prob(forget_loader, retain_model,device)
    return compute_metrics((shadow_train_prob, shadow_train_labels), (shadow_test_prob, shadow_test_labels), (target_test_prob, target_test_labels), (target_train_prob, target_train_labels)) 

if __name__ == "__main__":

    dataset = "celeba" #"cifar10"
    method = "labelflip" #"prune

    unlearnt_model_path =  f"neurips-submission/{method}_unlearn_{dataset}"#"neurips-submission/finetune_unlearn_cifar10"
    retrain_model_path = "neurips-submission/retrain_cifar10" if dataset == "cifar10" else "neurips-submission/retrain_celeba"
    #random_unlearnt_from_path =os.path.join(unlearnt_model_path,random.choice(os.listdir(unlearnt_model_path)))

    random_unlearnt_from_path =os.path.join(retrain_model_path,random.choice(os.listdir(retrain_model_path)))
    random_retrain_from_path = os.path.join(retrain_model_path,random.choice(os.listdir(retrain_model_path)))

    #random_unlearnt_from_path =  "neurips-submission/trainer_base_cifar.pt" if dataset == "cifar10" else "neurips-submission/trainer_base_celeba.pt"
    
    num_classes = 8 if dataset == "celeba" else 10
    args = dotmap.DotMap({"data":dotmap.DotMap({"BATCH_SIZE":512,"num_classes":num_classes,"num_workers":6}), "directory":dotmap.DotMap({"LOG_PATH":"./logs/"}),"device":"cuda:3"})
    train_loader,retain_loader, forget_loader, validation_loader,test_loader = get_celeba_dataloaders(args,balanced=False) if dataset == "celeba" else get_cifar10_dataloaders(args,balanced=False)
    
    unl_model = resnet18(num_classes=num_classes).to(args.device)
    retrain_model = resnet18(num_classes=num_classes).to(args.device)

    if "model_state_dict" in torch.load(random_unlearnt_from_path,map_location=args.device).keys():
        unl_model.load_state_dict(torch.load(random_unlearnt_from_path,map_location=args.device)["model_state_dict"])
    else:
        if method == "prune":
            unl_model.load_state_dict(torch.load(random_unlearnt_from_path,map_location=args.device))
        else:    
            unl_model.load_state_dict(torch.load(random_unlearnt_from_path,map_location=args.device))
    
    if "model_state_dict" in torch.load(random_retrain_from_path, map_location=args.device).keys():
        retrain_model.load_state_dict(torch.load(random_retrain_from_path, map_location=args.device)["model_state_dict"])
    else:
        retrain_model.load_state_dict(torch.load(random_retrain_from_path, map_location=args.device))

    #unl_model.load_state_dict(torch.load("/research/hal-gaudisac/unlearning/models/model_RLF_42_resnet18.pt",map_location=args.device)["model_state_dict"])
    #unl_model.load_state_dict(torch.load(random_retrain_from_path,map_location=args.device)["model_state_dict"])


    #SVC_retrain_unl(retain_loader, forget_loader, retrain_model, unl_model,device=args.device)
    SVC_MIA(retain_loader, None, forget_loader, test_loader, unl_model,device=args.device)


