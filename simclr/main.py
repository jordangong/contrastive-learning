import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable

import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision.transforms import transforms

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from libs.criteria import InfoNCELoss
from libs.datautils import color_distortion, Clip, RandomGaussianBlur, TwinTransform
from libs.optimizers import LARS
from libs.schedulers import LinearWarmupAndCosineAnneal, LinearLR
from libs.utils import Trainer, BaseConfig
from libs.logging import BaseBatchLogRecord, Loggers
from simclr.models import CIFARSimCLRResNet50, ImageNetSimCLRResNet50


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='Contrastive baseline SimCLR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--codename', default='cifar10-simclr-128-lars-warmup',
                        type=str, help="Model descriptor")
    parser.add_argument('--log-dir', default='logs', type=str,
                        help="Path to log directory")
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str,
                        help="Path to checkpoints directory")
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-iters', default=23438, type=int,
                        help='Number of iters (default is 50 epochs equiv., '
                             'around dataset_size * epochs / batch_size)')
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        help='Path to config file (optional)')

    # TODO: Add model hyperparams dataclass
    parser.add_argument('--hid-dim', default=2048, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--out-dim', default=128, type=int,
                        help='Number of dimension after projection')
    parser.add_argument('--temp', default=0.5, type=float,
                        help='Temperature in InfoNCE loss')

    dataset_group = parser.add_argument_group('Dataset parameters')
    dataset_group.add_argument('--dataset-dir', default='dataset', type=str,
                               help="Path to dataset directory")
    dataset_group.add_argument('--dataset', default='cifar10', type=str,
                               choices=('cifar10, cifar100', 'imagenet'),
                               help="Name of dataset")
    dataset_group.add_argument('--crop-size', default=32, type=int,
                               help='Random crop size after resize')
    dataset_group.add_argument('--crop-scale-range', nargs=2, default=(0.8, 1),
                               type=float, help='Random resize scale range',
                               metavar=('start', 'stop'))
    dataset_group.add_argument('--hflip-prob', default=0.5, type=float,
                               help='Random horizontal flip probability')
    dataset_group.add_argument('--distort-strength', default=0.5, type=float,
                               help='Distortion strength')
    dataset_group.add_argument('--gauss-ker-scale', default=10, type=float,
                               help='Gaussian kernel scale factor '
                                    '(s = img_size / ker_size)')
    dataset_group.add_argument('--gauss-sigma-range', nargs=2, default=(0.1, 2),
                               type=float, help='Random gaussian blur sigma range',
                               metavar=('start', 'stop'))
    dataset_group.add_argument('--gauss-prob', default=0.5, type=float,
                               help='Random gaussian blur probability')

    dataloader_group = parser.add_argument_group('Dataloader parameters')
    dataloader_group.add_argument('--batch-size', default=128, type=int,
                                  help='Batch size')
    dataloader_group.add_argument('--num-workers', default=2, type=int,
                                  help='Number of dataloader processes')

    optim_group = parser.add_argument_group('Optimizer parameters')
    optim_group.add_argument('--optim', default='lars', type=str,
                             choices=('adam', 'sgd', 'lars'),
                             help="Name of optimizer")
    optim_group.add_argument('--lr', default=1., type=float,
                             help='Learning rate')
    optim_group.add_argument('--betas', nargs=2, default=(0.9, 0.999), type=float,
                             help='Adam betas', metavar=('beta1', 'beta2'))
    optim_group.add_argument('--momentum', default=0.9, type=float,
                             help='SDG momentum')
    optim_group.add_argument('--weight-decay', default=1e-6, type=float,
                             help='Weight decay (l2 regularization)')

    sched_group = parser.add_argument_group('Optimizer parameters')
    sched_group.add_argument('--sched', default='warmup-anneal', type=str,
                             help="Name of scheduler")
    sched_group.add_argument('--warmup-iters', default=2344, type=int,
                             help='Epochs for warmup (`warmup-anneal` scheduler only)')

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
class SimCLRConfig(BaseConfig):
    @dataclass
    class DatasetConfig(BaseConfig.DatasetConfig):
        dataset_dir: str
        crop_size: int
        crop_scale_range: tuple[float, float]
        hflip_prob: float
        distort_strength: float
        gauss_ker_scale: float
        gauss_sigma_range: tuple[float, float]
        gauss_prob: float

    @dataclass
    class OptimConfig(BaseConfig.OptimConfig):
        momentum: float
        betas: tuple[float, float]
        weight_decay: float

    @dataclass
    class SchedConfig(BaseConfig.SchedConfig):
        sched: str | None
        warmup_iters: int


class SimCLRTrainer(Trainer):
    def __init__(self, hid_dim, out_dim, **kwargs):
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        super(SimCLRTrainer, self).__init__(**kwargs)

    @dataclass
    class BatchLogRecord(BaseBatchLogRecord):
        lr: float | None
        train_loss: float | None
        train_accuracy: float | None
        eval_loss: float | None
        eval_accuracy: float | None

    @staticmethod
    def _prepare_dataset(dataset_config: SimCLRConfig.DatasetConfig) -> tuple[Dataset, Dataset]:
        basic_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(
                dataset_config.crop_size,
                scale=dataset_config.crop_scale_range,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(dataset_config.hflip_prob),
            color_distortion(dataset_config.distort_strength),
        ])
        if dataset_config.dataset in {'cifar10', 'cifar100', 'cifar'}:
            transform = transforms.Compose([
                basic_augmentation,
                transforms.ToTensor(),
                Clip(),
            ])
            if dataset_config.dataset in {'cifar10', 'cifar'}:
                train_set = CIFAR10(dataset_config.dataset_dir, train=True,
                                    transform=TwinTransform(transform),
                                    download=True)
                test_set = CIFAR10(dataset_config.dataset_dir, train=False,
                                   transform=TwinTransform(transform))
            else:  # CIFAR-100
                train_set = CIFAR100(dataset_config.dataset_dir, train=True,
                                     transform=TwinTransform(transform),
                                     download=True)
                test_set = CIFAR100(dataset_config.dataset_dir, train=False,
                                    transform=TwinTransform(transform))
        elif dataset_config.dataset in {'imagenet1k', 'imagenet'}:
            random_gaussian_blur = RandomGaussianBlur(
                kernel_size=dataset_config.crop_size // dataset_config.gauss_ker_scale,
                sigma_range=dataset_config.gauss_sigma_range,
                p=dataset_config.gauss_prob
            ),
            transform = transforms.Compose([
                basic_augmentation,
                random_gaussian_blur,
                transforms.ToTensor(),
                Clip()
            ])
            train_set = ImageNet(dataset_config.dataset_dir, 'train',
                                 transform=TwinTransform(transform))
            test_set = ImageNet(dataset_config.dataset_dir, 'val',
                                transform=TwinTransform(transform))
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset_config.dataset}")

        return train_set, test_set

    def _init_models(self, dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        if dataset in {'cifar10', 'cifar100', 'cifar'}:
            model = CIFARSimCLRResNet50(self.hid_dim, self.out_dim)
        elif dataset in {'imagenet1k', 'imagenet'}:
            model = ImageNetSimCLRResNet50(self.hid_dim, self.out_dim)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset}")

        yield 'model', model

    @staticmethod
    def _configure_optimizers(
            models: Iterable[tuple[str, torch.nn.Module]],
            optim_config: SimCLRConfig.OptimConfig,
    ) -> Iterable[tuple[str, torch.optim.Optimizer]]:
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True
            if optim_config.optim == 'lars' and 'bias' in name:
                return True

        for model_name, model in models:
            param_groups = [
                {
                    'params': [p for name, p in model.named_parameters()
                               if not exclude_from_wd_and_adaptation(name)],
                    'weight_decay': optim_config.weight_decay,
                    'layer_adaptation': True,
                },
                {
                    'params': [p for name, p in model.named_parameters()
                               if exclude_from_wd_and_adaptation(name)],
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
            elif optim_config.optim in {'sgd', 'lars'}:
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
            sched_config: SimCLRConfig.SchedConfig,
    ) -> Iterable[tuple[str, torch.optim.lr_scheduler._LRScheduler]
                  | tuple[str, None]]:
        for optim_name, optim in optims:
            if sched_config.sched == 'warmup-anneal':
                scheduler = LinearWarmupAndCosineAnneal(
                    optim,
                    warm_up=sched_config.warmup_iters / num_iters,
                    T_max=num_iters,
                    last_epoch=last_iter,
                )
            elif sched_config.sched == 'linear':
                scheduler = LinearLR(
                    optim,
                    num_epochs=num_iters,
                    last_epoch=last_iter
                )
            elif sched_config.sched in {None, '', 'const'}:
                scheduler = None
            else:
                raise NotImplementedError(f"Unimplemented scheduler: '{sched_config.sched}'")

            yield f"{optim_name}_sched", scheduler

    def _custom_init_fn(self, config: SimCLRConfig):
        self.optims = {n: LARS(o) if config.optim_config.optim == 'lars' else o
                       for n, o in self.optims.items()}

    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        model = self.models['model']
        optim = self.optims['model_optim']
        sched = self.scheds['model_optim_sched']
        train_loader = iter(self.train_loader)
        model.train()
        for iter_ in range(self.restore_iter, num_iters):
            (input1, input2), _ = next(train_loader)
            input1, input2 = input1.to(device), input2.to(device)
            model.zero_grad()
            output1 = model(input1)
            output2 = model(input2)
            train_loss, train_accuracy = loss_fn(output1, output2)
            train_loss.backward()
            optim.step()
            self.log(logger, self.BatchLogRecord(
                iter_, num_iters, iter_, iter_, num_iters,
                optim.param_groups[0]['lr'],
                train_loss.item(), train_accuracy.item(),
                eval_loss=None, eval_accuracy=None,
            ))
            if (iter_ + 1) % (num_iters // 100) == 0:
                metrics = torch.Tensor(list(self.eval(loss_fn, device))).mean(0)
                eval_loss = metrics[0].item()
                eval_accuracy = metrics[1].item()
                eval_log = self.BatchLogRecord(
                    iter_, num_iters, iter_, iter_, num_iters,
                    lr=None, train_loss=None, train_accuracy=None,
                    eval_loss=eval_loss, eval_accuracy=eval_accuracy,
                )
                self.log(logger, eval_log)
                self.save_checkpoint(eval_log)
                model.train()
            if sched is not None:
                sched.step()

    def eval(self, loss_fn: Callable, device: torch.device):
        model = self.models['model']
        model.eval()
        with torch.no_grad():
            for (input1, input2), _ in self.test_loader:
                input1, input2 = input1.to(device), input2.to(device)
                output1 = model(input1)
                output2 = model(input2)
                loss, accuracy = loss_fn(output1, output2)
                yield loss.item(), accuracy.item()


if __name__ == '__main__':
    args = parse_args_and_config()
    config = SimCLRConfig.from_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SimCLRTrainer(
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        inf_mode=True,
        num_iters=args.num_iters,
        config=config,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
    )

    loggers = trainer.init_logger(args.log_dir)
    trainer.train(args.num_iters, InfoNCELoss(args.temp), loggers, device)
