from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from libs.logging import Loggers
from libs.utils import Trainer, BaseConfig


class PosReconTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(PosReconTrainer, self).__init__(*args, **kwargs)

    @staticmethod
    def _prepare_dataset(dataset_config: BaseConfig.DatasetConfig) -> tuple[Dataset, Dataset]:
        pass

    @staticmethod
    def _init_models(dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        pass

    @staticmethod
    def _configure_optimizers(models: Iterable[tuple[str, torch.nn.Module]], optim_config: BaseConfig.OptimConfig) -> \
            Iterable[tuple[str, torch.optim.Optimizer]]:
        pass

    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        pass

    def eval(self, loss_fn: Callable, device: torch.device):
        pass
