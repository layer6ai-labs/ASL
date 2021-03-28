import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        return first_term + second_term
