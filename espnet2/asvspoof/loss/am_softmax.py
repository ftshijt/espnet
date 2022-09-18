import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class AMSoftmax(AbsASVSpoofLoss):
    def __init__(self, enc_dim, s=20, m=0.5):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(2, enc_dim))
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, emb: torch.Tensor, pred: torch.Tensor, label: torch.Tensor):
        batch_size = emb.shape[0]
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, 2)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot)
        label = label.type(torch.int64)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)
        loss = self.loss(self.sigmoid(margin_logits[:, 0]), label.view(-1).float())

        return loss

    def score(self, emb: torch.Tensor, pred: torch.Tensor):
        """Forward.
        Args:
            emb (torch.Tensor): embedding [Batch, emb_dim]
            pred  (torch.Tensor): prediction probability [Batch, 2]
        """
        batch_size = emb.shape[0]
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        return self.sigmoid(logits[:, 0])
