import warnings

import numpy as np
import torch


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]
