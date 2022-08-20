import argparse
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable

import sys
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision.transforms import transforms

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from libs.optimizers import LARS
from libs.logging import Loggers, BaseBatchLogRecord, BaseEpochLogRecord
from libs.utils import BaseConfig
from simclr.main import SimCLRTrainer, SimCLRConfig
from simclr.models import CIFARSimCLRResNet50, ImageNetSimCLRResNet50, CIFARSimCLRViTTiny


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='Contrastive baseline SimCLR (evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--codename', default='cifar10-simclr-linear',
                        type=str, help="Model descriptor")
    parser.add_argument('--log-dir', default='logs', type=str,
                        help="Path to log directory")
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str,
                        help="Path to checkpoints directory")
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-iters', default=50, type=int,
                        help='Number of iters')
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        help='Path to config file (optional)')

    # TODO: Add model hyperparams dataclass
    parser.add_argument('--encoder', default='resnet', type=str,
                        choices=('resnet', 'vit'),
                        help='Backbone of encoder')
    parser.add_argument('--hid-dim', default=2048, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--out-dim', default=128, type=int,
                        help='Number of dimension after projection')
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                        help='Pretrained checkpoint location')
    parser.add_argument('--finetune', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Finetune backbone or linear head only')

    dataset_group = parser.add_argument_group('Dataset parameters')
    dataset_group.add_argument('--dataset-dir', default='dataset', type=str,
                               help="Path to dataset directory")
    dataset_group.add_argument('--dataset', default='cifar10', type=str,
                               choices=('cifar10, cifar100', 'imagenet'),
                               help="Name of dataset")
    dataset_group.add_argument('--train-size', default=32, type=int,
                               help='Resize during training')
    dataset_group.add_argument('--test-size', default=32, type=int,
                               help='Resize during testing')
    dataset_group.add_argument('--test-crop-size', default=32, type=int,
                               help='Center crop size during testing')
    dataset_group.add_argument('--hflip-prob', default=0.5, type=float,
                               help='Random horizontal flip probability')

    dataloader_group = parser.add_argument_group('Dataloader parameters')
    dataloader_group.add_argument('--batch-size', default=256, type=int,
                                  help='Batch size')
    dataloader_group.add_argument('--num-workers', default=2, type=int,
                                  help='Number of dataloader processes')

    optim_group = parser.add_argument_group('Optimizer parameters')
    optim_group.add_argument('--optim', default='sgd', type=str,
                             choices=('adam', 'sgd', 'lars'),
                             help="Name of optimizer")
    optim_group.add_argument('--lr', default=1e-3, type=float,
                             help='Learning rate')
    optim_group.add_argument('--betas', nargs=2, default=(0.9, 0.999), type=float,
                             help='Adam betas', metavar=('beta1', 'beta2'))
    optim_group.add_argument('--momentum', default=0.9, type=float,
                             help='SDG momentum')
    optim_group.add_argument('--weight-decay', default=0., type=float,
                             help='Weight decay (l2 regularization)')

    sched_group = parser.add_argument_group('Scheduler parameters')
    sched_group.add_argument('--sched', default=None, type=str,
                             choices=('const', None, 'linear', 'warmup-anneal'),
                             help="Name of scheduler")
    sched_group.add_argument('--warmup-iters', default=5, type=int,
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
class SimCLREvalConfig(SimCLRConfig):
    @dataclass
    class DatasetConfig(BaseConfig.DatasetConfig):
        dataset_dir: str
        train_size: int
        test_size: int
        test_crop_size: int
        hflip_prob: float


class SimCLREvalTrainer(SimCLRTrainer):
    def __init__(self, pretrained_checkpoint, finetune, **kwargs):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.finetune = finetune
        super(SimCLREvalTrainer, self).__init__(**kwargs)

    @dataclass
    class BatchLogRecord(BaseBatchLogRecord):
        lr: float
        train_loss: float

    @dataclass
    class EpochLogRecord(BaseEpochLogRecord):
        eval_loss: float
        eval_accuracy: float

    @staticmethod
    def _prepare_dataset(dataset_config: SimCLREvalConfig.DatasetConfig) -> tuple[Dataset, Dataset]:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(dataset_config.hflip_prob),
            transforms.Resize(dataset_config.train_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(dataset_config.test_size),
            transforms.CenterCrop(dataset_config.test_crop_size),
            transforms.ToTensor(),
        ])
        if dataset_config.dataset in {'cifar10', 'cifar100', 'cifar'}:
            if dataset_config.dataset in {'cifar10', 'cifar'}:
                train_set = CIFAR10(dataset_config.dataset_dir, train=True,
                                    transform=train_transform, download=True)
                test_set = CIFAR10(dataset_config.dataset_dir, train=False,
                                   transform=test_transform)
            else:  # CIFAR-100
                train_set = CIFAR100(dataset_config.dataset_dir, train=True,
                                     transform=train_transform, download=True)
                test_set = CIFAR100(dataset_config.dataset_dir, train=False,
                                    transform=test_transform)
        elif dataset_config.dataset in {'imagenet1k', 'imagenet'}:
            train_set = ImageNet(dataset_config.dataset_dir, 'train',
                                 transform=train_transform)
            test_set = ImageNet(dataset_config.dataset_dir, 'val',
                                transform=test_transform)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset_config.dataset}")

        return train_set, test_set

    def _init_models(self, dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        if dataset in {'cifar10', 'cifar100', 'cifar'}:
            if self.encoder == 'resnet':
                backbone = CIFARSimCLRResNet50(self.hid_dim, pretrain=False)
            elif self.encoder == 'vit':
                backbone = CIFARSimCLRViTTiny(self.hid_dim, pretrain=False)
            else:
                raise NotImplementedError(f"Unimplemented encoder: '{self.encoder}")
            if dataset in {'cifar10', 'cifar'}:
                classifier = torch.nn.Linear(self.hid_dim, 10)
            else:
                classifier = torch.nn.Linear(self.hid_dim, 100)
        elif dataset in {'imagenet1k', 'imagenet'}:
            if self.encoder == 'resnet':
                backbone = ImageNetSimCLRResNet50(self.hid_dim, pretrain=False)
            else:
                raise NotImplementedError(f"Unimplemented encoder: '{self.encoder}")
            classifier = torch.nn.Linear(self.hid_dim, 1000)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset}")

        yield 'backbone', backbone
        yield 'classifier', classifier

    def _custom_init_fn(self, config: SimCLREvalConfig):
        self.optims = {n: LARS(o) if config.optim_config.optim == 'lars' else o
                       for n, o in self.optims.items()}
        if self.restore_iter == 0:
            pretrained_checkpoint = torch.load(self.pretrained_checkpoint)
            backbone_checkpoint = pretrained_checkpoint['model_state_dict']
            backbone_state_dict = {k: v for k, v in backbone_checkpoint.items()
                                   if k in self.models['backbone'].state_dict()}
            self.models['backbone'].load_state_dict(backbone_state_dict)

    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        backbone, classifier = self.models.values()
        optim_b, optim_c = self.optims.values()
        sched_b, sched_c = self.scheds.values()
        loader_size = len(self.train_loader)
        num_batches = num_iters * loader_size
        for iter_ in range(self.restore_iter, num_iters):
            if self.finetune:
                backbone.train()
            else:
                backbone.eval()
            classifier.train()
            for batch, (images, targets) in enumerate(self.train_loader):
                global_batch = iter_ * loader_size + batch
                images, targets = images.to(device), targets.to(device)
                classifier.zero_grad()
                if self.finetune:
                    backbone.zero_grad()
                    embedding = backbone(images)
                else:
                    with torch.no_grad():
                        embedding = backbone(images)
                logits = classifier(embedding)
                train_loss = loss_fn(logits, targets)
                train_loss.backward()
                if self.finetune:
                    optim_b.step()
                optim_c.step()
                self.log(logger, self.BatchLogRecord(
                    batch, num_batches, global_batch, iter_, num_iters,
                    optim_c.param_groups[0]['lr'], train_loss.item()
                ))
            if (iter_ + 1) % (num_iters // 10) == 0:
                metrics = torch.Tensor(list(self.eval(loss_fn, device))).mean(0)
                eval_loss = metrics[0].item()
                eval_accuracy = metrics[1].item()
                epoch_log = self.EpochLogRecord(iter_, num_iters,
                                                eval_loss, eval_accuracy)
                self.log(logger, epoch_log)
                self.save_checkpoint(epoch_log)
            if sched_b is not None and self.finetune:
                sched_b.step()
            if sched_c is not None:
                sched_c.step()

    def eval(self, loss_fn: Callable, device: torch.device):
        backbone, classifier = self.models.values()
        backbone.eval()
        classifier.eval()
        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(device), targets.to(device)
                embedding = backbone(images)
                logits = classifier(embedding)
                loss = loss_fn(logits, targets)
                prediction = logits.argmax(1)
                accuracy = (prediction == targets).float().mean()
                yield loss.item(), accuracy.item()


if __name__ == '__main__':
    args = parse_args_and_config()
    config = SimCLREvalConfig.from_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SimCLREvalTrainer(
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        inf_mode=False,
        num_iters=args.num_iters,
        config=config,
        encoder=args.encoder,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetune=args.finetune,
    )

    loggers = trainer.init_logger(args.log_dir)
    trainer.train(args.num_iters, torch.nn.CrossEntropyLoss(), loggers, device)
