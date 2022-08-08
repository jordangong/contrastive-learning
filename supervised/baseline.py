import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable

import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from libs.datautils import Clip
from libs.schedulers import LinearLR
from libs.utils import Trainer, BaseConfig
from libs.logging import BaseBatchLogRecord, BaseEpochLogRecord, Loggers
from models import CIFARResNet50


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='Supervised baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--codename', default='cifar10-resnet50-256-adam-linear',
                        type=str, help="Model descriptor")
    parser.add_argument('--log-dir', default='logs', type=str,
                        help="Path to log directory")
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str,
                        help="Path to checkpoints directory")
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-iters', default=1000, type=int,
                        help='Number of iters (epochs)')
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        help='Path to config file (optional)')

    dataset_group = parser.add_argument_group('Dataset parameters')
    dataset_group.add_argument('--dataset-dir', default='dataset', type=str,
                               help="Path to dataset directory")
    dataset_group.add_argument('--dataset', default='cifar10', type=str,
                               choices=('cifar', 'cifar10, cifar100'),
                               help="Name of dataset")
    dataset_group.add_argument('--crop-size', default=32, type=int,
                               help='Random crop size after resize')
    dataset_group.add_argument('--crop-scale-range', nargs=2, default=(0.8, 1),
                               type=float, help='Random resize scale range',
                               metavar=('start', 'stop'))
    dataset_group.add_argument('--hflip-prob', default=0.5, type=float,
                               help='Random horizontal flip probability')

    dataloader_group = parser.add_argument_group('Dataloader parameters')
    dataloader_group.add_argument('--batch-size', default=256, type=int,
                                  help='Batch size')
    dataloader_group.add_argument('--num-workers', default=2, type=int,
                                  help='Number of dataloader processes')

    optim_group = parser.add_argument_group('Optimizer parameters')
    optim_group.add_argument('--optim', default='adam', type=str,
                             choices=('adam', 'sgd'), help="Name of optimizer")
    optim_group.add_argument('--lr', default=1e-3, type=float,
                             help='Learning rate')
    optim_group.add_argument('--betas', nargs=2, default=(0.9, 0.999), type=float,
                             help='Adam betas', metavar=('beta1', 'beta2'))
    optim_group.add_argument('--momentum', default=0.9, type=float,
                             help='SDG momentum')
    optim_group.add_argument('--weight-decay', default=1e-6, type=float,
                             help='Weight decay (l2 regularization)')

    sched_group = parser.add_argument_group('Optimizer parameters')
    sched_group.add_argument('--sched', default='linear', type=str,
                             choices=(None, '', 'linear'), help="Name of scheduler")

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(args.config)
        args.__dict__ |= {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in config.items()
        }
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.codename)
    args.log_dir = os.path.join(args.log_dir, args.codename)

    return args


@dataclass
class SupBaselineConfig(BaseConfig):
    @dataclass
    class DatasetConfig(BaseConfig.DatasetConfig):
        dataset_dir: str
        crop_size: int
        crop_scale_range: tuple[float, float]
        hflip_prob: float

    @dataclass
    class OptimConfig(BaseConfig.OptimConfig):
        momentum: float | None
        betas: tuple[float, float] | None
        weight_decay: float

    @dataclass
    class SchedConfig(BaseConfig.SchedConfig):
        sched: str | None


class SupBaselineTrainer(Trainer):
    def __init__(self, **kwargs):
        super(SupBaselineTrainer, self).__init__(**kwargs)

    @dataclass
    class BatchLogRecord(BaseBatchLogRecord):
        lr: float
        train_loss: float

    @dataclass
    class EpochLogRecord(BaseEpochLogRecord):
        eval_loss: float
        eval_accuracy: float

    @staticmethod
    def _prepare_dataset(dataset_config: SupBaselineConfig.DatasetConfig) -> tuple[Dataset, Dataset]:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                dataset_config.crop_size,
                scale=dataset_config.crop_scale_range,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(dataset_config.hflip_prob),
            transforms.ToTensor(),
            Clip(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if dataset_config.dataset in {'cifar10', 'cifar'}:
            train_set = CIFAR10(dataset_config.dataset_dir, train=True,
                                transform=train_transform, download=True)
            test_set = CIFAR10(dataset_config.dataset_dir, train=False,
                               transform=test_transform)
        elif dataset_config.dataset == 'cifar100':
            train_set = CIFAR100(dataset_config.dataset_dir, train=True,
                                 transform=train_transform, download=True)
            test_set = CIFAR100(dataset_config.dataset_dir, train=False,
                                transform=test_transform)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset_config.dataset}")

        return train_set, test_set

    @staticmethod
    def _init_models(dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        if dataset in {'cifar10', 'cifar'}:
            model = CIFARResNet50(num_classes=10)
        elif dataset == 'cifar100':
            model = CIFARResNet50(num_classes=100)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset}")

        yield 'model', model

    @staticmethod
    def _configure_optimizers(
            models: Iterable[tuple[str, torch.nn.Module]],
            optim_config: SupBaselineConfig.OptimConfig,
    ) -> Iterable[tuple[str, torch.optim.Optimizer]]:
        for model_name, model in models:
            param_groups = [
                {
                    'params': [p for name, p in model.named_parameters()
                               if 'bn' not in name],
                    'weight_decay': optim_config.weight_decay,
                    'layer_adaptation': True,
                },
                {
                    'params': [p for name, p in model.named_parameters()
                               if 'bn' in name],
                    'weight_decay': 0.,
                    'layer_adaptation': False,
                },
            ]
            if optim_config.optim == 'adam':
                optimizer = torch.optim.Adam(
                    param_groups,
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                )
            elif optim_config.optim == 'sgd':
                optimizer = torch.optim.SGD(
                    param_groups,
                    lr=optim_config.lr,
                    momentum=optim_config.momentum,
                )
            else:
                raise NotImplementedError(f"Unimplemented optimizer: '{optim_config.optim}'")

            yield f"{model_name}_optim", optimizer

    @staticmethod
    def _configure_scheduler(
            optims: Iterable[tuple[str, torch.optim.Optimizer]],
            last_iter: int,
            num_iters: int,
            sched_config: SupBaselineConfig.SchedConfig
    ) -> Iterable[tuple[str, torch.optim.lr_scheduler._LRScheduler]
                  | tuple[str, None]]:
        for optim_name, optim in optims:
            if sched_config.sched == 'linear':
                sched = LinearLR(optim, num_iters, last_epoch=last_iter)
            elif sched_config.sched is None:
                sched = None
            else:
                raise NotImplementedError(f"Unimplemented scheduler: {sched_config.sched}")
            yield f"{optim_name}_sched", sched

    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        model = self.models['model']
        optim = self.optims['model_optim']
        sched = self.scheds['model_optim_sched']
        loader_size = len(self.train_loader)
        num_batches = num_iters * loader_size
        for iter_ in range(self.restore_iter, num_iters):
            model.train()
            for batch, (images, targets) in enumerate(self.train_loader):
                global_batch = iter_ * loader_size + batch
                images, targets = images.to(device), targets.to(device)
                model.zero_grad()
                output = model(images)
                train_loss = loss_fn(output, targets)
                train_loss.backward()
                optim.step()
                self.log(logger, self.BatchLogRecord(
                    batch, num_batches, global_batch, iter_, num_iters,
                    optim.param_groups[0]['lr'], train_loss.item()
                ))
            metrics = torch.Tensor(list(self.eval(loss_fn, device))).mean(0)
            eval_loss = metrics[0].item()
            eval_accuracy = metrics[1].item()
            epoch_log = self.EpochLogRecord(iter_, num_iters, eval_loss, eval_accuracy)
            self.log(logger, epoch_log)
            self.save_checkpoint(epoch_log)
            # Step after save checkpoint, otherwise the schedular will one iter ahead after restore
            if sched is not None:
                sched.step()

    def eval(self, loss_fn: Callable, device: torch.device):
        model = self.models['model']
        model.eval()
        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                loss = loss_fn(output, targets)
                prediction = output.argmax(1)
                accuracy = (prediction == targets).float().mean()
                yield loss.item(), accuracy.item()


if __name__ == '__main__':
    args = parse_args_and_config()
    config = SupBaselineConfig.from_args(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainer = SupBaselineTrainer(
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        inf_mode=False,
        num_iters=args.num_iters,
        config=config,
    )

    loggers = trainer.init_logger(args.log_dir)
    trainer.train(args.num_iters, torch.nn.CrossEntropyLoss(), loggers, device)
