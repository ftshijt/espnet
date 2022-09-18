import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class OCSoftmax(AbsASVSpoofLoss):
    def __init__(self, enc_dim, m_real=0.5, m_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = enc_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, emb: torch.Tensor, pred: torch.Tensor, label: torch.Tensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(emb, p=2, dim=1)

        scores = x @ w.transpose(0,1)

        scores[label == 0] = self.m_real - scores[label == 0]
        scores[label == 1] = scores[label == 1] - self.m_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss

    def score(self, emb: torch.Tensor, pred: torch.Tensor):
        """Forward.
        Args:
            emb (torch.Tensor): embedding [Batch, emb_dim]
            pred  (torch.Tensor): prediction probability [Batch, 2]
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(emb, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        return scores.squeeze(1)

