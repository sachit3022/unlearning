import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from celeba_dataset import get_dataset,UnlearnCelebADataset
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader,Subset


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
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

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model,device):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model,device)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model,device)

    target_train_prob, target_train_labels = collect_prob(target_train, model,device)
    target_test_prob, target_test_labels = collect_prob(target_test, model,device)

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

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    if target_train is not None:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test is not None:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr

    acc_corr = SVC_fit_predict(
        shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    acc_entr = SVC_fit_predict(
        shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    )
    acc_m_entr = SVC_fit_predict(
        shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    )
    acc_prob = SVC_fit_predict(
        shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    )
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    print(m)
    return m

if __name__ == "__main__":
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=8).to(device)
    model.load_state_dict(torch.load("models/model_scratch_42_resnet18.pt",map_location=device)["model_state_dict"])

    train_dataset = UnlearnCelebADataset("train")
    retain_dataset = UnlearnCelebADataset("retain")
    forget_dataset = UnlearnCelebADataset("forget")
    valid_dataset = UnlearnCelebADataset("valid")
    test_dataset = UnlearnCelebADataset("test")

    retain_loader_train = DataLoader(Subset(train_dataset,range(0,len(test_dataset))),batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    forget_loader = DataLoader(forget_dataset,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    SVC_MIA(shadow_train = retain_loader_train,shadow_test = test_loader, target_train = None, target_test = forget_loader, model=model,device=device)
