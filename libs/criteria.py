import torch
from torch import nn, Tensor
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.01):
        super().__init__()
        self.temp = temp

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        bz = feature.size(0) // 2
        feat_norm = F.normalize(feature)
        feat1_norm, feat2_norm = feat_norm.split(bz)
        logits = feat1_norm @ feat2_norm.T
        pos_logits_mask = torch.eye(bz, dtype=torch.bool)
        pos_logits = logits[pos_logits_mask].unsqueeze(-1)
        neg_logits = logits[~pos_logits_mask].view(bz, -1)
        # Put the positive at first (0-th) and maximize its likelihood
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(bz, dtype=torch.long, device=feature.device)
        loss_contra = F.cross_entropy(logits / self.temp, labels)
        acc_contra = (logits.argmax(dim=1) == labels).float().mean()

        return loss_contra, acc_contra
