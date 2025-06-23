import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from balanced_loss import Loss as Focal

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :],
                               index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def build_loss(cfg, samples_per_class, device):
    if cfg.loss.type=="CE":
        return nn.CrossEntropyLoss()
    if cfg.loss.type=="FL":
        return Focal(
            loss_type="focal_loss",
            beta=cfg.loss.fl_beta,
            fl_gamma=cfg.loss.fl_gamma,
            samples_per_class=samples_per_class,
            class_balanced=True
        )
    if cfg.loss.type=="LDAM":
        effective_num = 1.0 - np.power(0.9999, samples_per_class)
        _weight = (1.0 - 0.9999) / np.array(effective_num)
        _weight = _weight / np.sum(_weight) * len(samples_per_class)
        weights = torch.tensor(_weight).float()
        return LDAMLoss(cls_num_list=samples_per_class, weight=torch.tensor(weights).to(device))
    raise ValueError

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, max=10))

def softplus_evidence(y):
    return F.softplus(y)
    
def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(alpha, labels, epoch, annealing_step, num_cls, reduction=False, device="cuda:0"):
    #kl_weight=0.01
    #evidence = exp_evidence(outputs)
    #alpha = evidence + 1
    # Evidence-based loss (Malinin & Gales)
    S = torch.sum(alpha, dim=1, keepdim=True)
    one_hot = F.one_hot(labels, num_classes=num_cls).float()
    # Expected log-likelihood term
    loglik = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # Annealing coefficient (optional)
    if epoch is None:
        annealing_coef = torch.tensor(1.0, dtype=torch.float32)
    else:
        annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch / annealing_step, dtype=torch.float32),
    )
    # KL regularization
    kl_alpha = (alpha - 1) * (1 - one_hot) + 1
    kl = kl_divergence(kl_alpha, num_cls, device)
    kl_div = annealing_coef * kl
    
    #loss = loglik + kl_div * kl_weight
    loss = loglik + kl_div
    
    if reduction == True:
        return torch.mean(loss)
    else:
        return loss.squeeze()

def combined_loss_switch(criterion, epoch, alpha, logits, y, num_classes, reduction, device, switch_epoch=10):
    """
        Si epoch < switch_epoch → focal loss seule
        Sinon → combinaison EDL + KL
    """
    if epoch==None:
        return edl_loss(alpha, y, epoch, 10, num_classes,reduction=reduction, device=device)
    elif epoch < switch_epoch:
        return criterion(logits,y)
    else:
        return edl_loss(alpha, y, epoch, 10, num_classes,reduction=reduction, device=device)

def combined_loss(criterion, epoch, total_epochs, alpha, logits, y, num_classes, reduction, device):
    if epoch==None:
        return edl_loss(alpha, y, epoch, 10, num_classes,reduction=reduction, device=device)
    else:
        #lam = min(epoch / total_epochs, 1.0)
        lam = 0.0
        print(f"########### lam fl : {lam}###########")
        ce_loss = criterion(logits,y)
        el_loss = edl_loss(alpha, y, epoch, 10, num_classes,reduction=reduction, device=device)
        total_loss = el_loss + lam * ce_loss
        return total_loss
                            
