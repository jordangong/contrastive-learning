import dataclasses
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Callable

import torch
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from libs.logging import CSV_EPOCH_LOGGER, CSV_BATCH_LOGGER, BaseBatchLogRecord, BaseEpochLogRecord, Loggers, \
    init_csv_logger, csv_logger, tensorboard_logger


@dataclass
class BaseConfig:
    @dataclass
    class DatasetConfig:
        dataset: str

    @dataclass
    class DataLoaderConfig:
        batch_size: int
        num_workers: int

    @dataclass
    class OptimConfig:
        optim: str
        lr: float

    @dataclass
    class SchedConfig:
        sched: None

    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    optim_config: OptimConfig
    sched_config: SchedConfig

    @staticmethod
    def _config_from_args(args, dcls):
        return dcls(**{f.name: getattr(args, f.name)
                       for f in dataclasses.fields(dcls)})

    @classmethod
    def from_args(cls, args):
        dataset_config = cls._config_from_args(args, cls.DatasetConfig)
        dataloader_config = cls._config_from_args(args, cls.DataLoaderConfig)
        optim_config = cls._config_from_args(args, cls.OptimConfig)
        sched_config = cls._config_from_args(args, cls.SchedConfig)

        return cls(dataset_config, dataloader_config, optim_config, sched_config)


class Trainer(ABC):
    def __init__(
            self,
            seed: int,
            checkpoint_dir: str,
            device: torch.device,
            inf_mode: bool,
            num_iters: int,
            config: BaseConfig,
    ):
        self._args = locals()
        self._set_seed(seed)

        train_set, test_set = self._prepare_dataset(config.dataset_config)
        train_loader, test_loader = self._create_dataloader(
            train_set, test_set, inf_mode, config.dataloader_config
        )

        models = self._init_models(config.dataset_config.dataset)
        models = {n: m.to(device) for n, m in models}
        optims = dict(self._configure_optimizers(models.items(), config.optim_config))
        last_metrics = self._auto_load_checkpoint(
            checkpoint_dir, inf_mode, **(models | optims)
        )

        if last_metrics is None:
            last_iter = -1
        elif isinstance(last_metrics, BaseEpochLogRecord):
            last_iter = last_metrics.epoch
        elif isinstance(last_metrics, BaseBatchLogRecord):
            last_iter = last_metrics.global_batch
        else:
            raise NotImplementedError(f"Unknown log type: '{type(last_metrics)}'")
        if not inf_mode:
            num_iters *= len(train_loader)
        scheds = dict(self._configure_scheduler(
            optims.items(), last_iter, num_iters, config.sched_config,
        ))

        self._custom_init_fn(config)

        self.restore_iter = last_iter + 1
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.models = models
        self.optims = optims
        self.scheds = scheds
        self._inf_mode = inf_mode
        self._checkpoint_dir = checkpoint_dir

    @dataclass
    class BatchLogRecord(BaseBatchLogRecord):
        pass

    @dataclass
    class EpochLogRecord(BaseEpochLogRecord):
        pass

    @staticmethod
    def _set_seed(seed):
        if seed in {-1, None, ''}:
            cudnn.benchmark = True
        else:
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True

    def init_logger(self, log_dir):
        csv_batch_log_fname = os.path.join(log_dir, 'batch-log.csv')
        csv_batch_logger = init_csv_logger(
            name=CSV_BATCH_LOGGER,
            filename=csv_batch_log_fname,
            metric_names=[f.name for f in dataclasses.fields(self.BatchLogRecord)]
        )
        csv_epoch_logger = None
        if not self._inf_mode:
            csv_epoch_log_fname = os.path.join(log_dir, 'epoch-log.csv')
            csv_epoch_logger = init_csv_logger(
                name=CSV_EPOCH_LOGGER,
                filename=csv_epoch_log_fname,
                metric_names=[f.name for f in dataclasses.fields(self.EpochLogRecord)]
            )
        tb_logger = SummaryWriter(os.path.join(log_dir, 'runs'))

        return Loggers(csv_batch_logger, csv_epoch_logger, tb_logger)

    def dump_args(self, exclude=frozenset()) -> dict:
        return {k: v for k, v in self._args.items() if k not in {'self'} | exclude}

    @staticmethod
    @abstractmethod
    def _prepare_dataset(dataset_config: BaseConfig.DatasetConfig) -> tuple[Dataset, Dataset]:
        train_set = Dataset()
        test_set = Dataset()
        return train_set, test_set

    @staticmethod
    def _create_dataloader(
            train_set: Dataset, test_set: Dataset,
            inf_mode: bool, dataloader_config: BaseConfig.DataLoaderConfig
    ) -> tuple[DataLoader, DataLoader]:
        if inf_mode:
            inf_sampler = RandomSampler(train_set,
                                        replacement=True,
                                        num_samples=int(1e20))
            train_loader = DataLoader(train_set,
                                      sampler=inf_sampler,
                                      batch_size=dataloader_config.batch_size,
                                      num_workers=dataloader_config.num_workers)
        else:
            train_loader = DataLoader(train_set,
                                      shuffle=True,
                                      batch_size=dataloader_config.batch_size,
                                      num_workers=dataloader_config.num_workers)
        test_loader = DataLoader(test_set,
                                 shuffle=False,
                                 batch_size=dataloader_config.batch_size,
                                 num_workers=dataloader_config.num_workers)

        return train_loader, test_loader

    @staticmethod
    @abstractmethod
    def _init_models(dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        model = torch.nn.Module()
        yield 'model_name', model

    @staticmethod
    @abstractmethod
    def _configure_optimizers(
            models: Iterable[tuple[str, torch.nn.Module]],
            optim_config: BaseConfig.OptimConfig
    ) -> Iterable[tuple[str, torch.optim.Optimizer]]:
        for model_name, model in models:
            optim = torch.optim.Optimizer([model.state_dict()], {})
            yield f"{model_name}_optim", optim

    def _auto_load_checkpoint(
            self,
            checkpoint_dir: str,
            inf_mode: bool,
            **modules
    ) -> None | BaseEpochLogRecord | BaseEpochLogRecord:
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoint_files = os.listdir(checkpoint_dir)
        if not checkpoint_files:
            return None
        iter2checkpoint = {int(os.path.splitext(checkpoint_file)[0]): checkpoint_file
                           for checkpoint_file in checkpoint_files}
        restore_iter = max(iter2checkpoint.keys())
        latest_checkpoint = iter2checkpoint[restore_iter]
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        for module_name in modules.keys():
            module_state_dict = checkpoint[f"{module_name}_state_dict"]
            modules[module_name].load_state_dict(module_state_dict)

        last_metrics = {k: v for k, v in checkpoint.items()
                        if not k.endswith('state_dict')}
        if inf_mode:
            last_metrics = self.BatchLogRecord(**last_metrics)
        else:
            last_metrics = self.EpochLogRecord(**last_metrics)

        return last_metrics

    @staticmethod
    @abstractmethod
    def _configure_scheduler(
            optims: Iterable[tuple[str, torch.optim.Optimizer]],
            last_iter: int, num_iters: int, sched_config: BaseConfig.SchedConfig,
    ) -> Iterable[tuple[str, torch.optim.lr_scheduler._LRScheduler]
                  | tuple[str, None]]:
        for optim_name, optim in optims:
            sched = torch.optim.lr_scheduler._LRScheduler(optim, -1)
            yield f"{optim_name}_sched", sched

    def _custom_init_fn(self, config: BaseConfig):
        pass

    @staticmethod
    @csv_logger
    @tensorboard_logger
    def log(loggers: Loggers, metrics: BaseBatchLogRecord | BaseEpochLogRecord):
        return loggers, metrics

    def save_checkpoint(self, metrics: BaseEpochLogRecord | BaseBatchLogRecord):
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        checkpoint_name = os.path.join(self._checkpoint_dir, f"{metrics.epoch:06d}.pt")
        models_state_dict = {f"{model_name}_state_dict": model.state_dict()
                             for model_name, model in self.models.items()}
        optims_state_dict = {f"{optim_name}_state_dict": optim.state_dict()
                             for optim_name, optim in self.optims.items()}
        checkpoint = metrics.__dict__ | models_state_dict | optims_state_dict
        torch.save(checkpoint, checkpoint_name)

    @abstractmethod
    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        pass

    @abstractmethod
    def eval(self, loss_fn: Callable, device: torch.device):
        pass
