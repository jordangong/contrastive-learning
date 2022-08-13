import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn, Tensor
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.01):
        super().__init__()
        self.temp = temp

    @staticmethod
    def _norm_and_stack(feat: Tensor) -> Tensor:
        local_feat_norm = F.normalize(feat)
        local_feat_norm_stack = torch.stack(local_feat_norm.chunk(2))

        return local_feat_norm_stack

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        feat_norm = torch.cat([
            rpc.rpc_sync(f"worker{i}", self._norm_and_stack, (feature,))
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
