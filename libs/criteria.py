import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn, Tensor
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.01):
        super().__init__()
        self.temp = temp
        self.local_feat_norm = None

    def get_local_feat_norm(self):
        return self.local_feat_norm

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        local_feat_norm = F.normalize(feature)
        self.local_feat_norm = torch.stack(local_feat_norm.chunk(2))
        feat_norm = torch.cat([
            rpc.rpc_sync(f"worker{i}", self.get_local_feat_norm)
            for i in range(dist.get_world_size())
        ], dim=1)
        bz = feat_norm.size(1)

        feat1_norm, feat2_norm = feat_norm[0], feat_norm[1]
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
