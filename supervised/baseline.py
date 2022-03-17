import argparse
import os
import random

import torch
import yaml
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import transforms, InterpolationMode

from datautils import color_distortion, Clip, RandomGaussianBlur
from models import CIFARResNet50, ImageNetResNet50
from optimizers import LARS
from schedulers import LinearWarmupAndCosineAnneal, LinearLR
from utils import training_log, setup_logging, EPOCH_LOGGER, BATCH_LOGGER


def build_parser():
    def range_parser(range_string: str):
        try:
            range_ = tuple(map(float, range_string.split('-')))
            return range_
        except:
            raise argparse.ArgumentTypeError("Range must be 'start-end.'")

    def merge_yaml(args):
        if args.config:
            config = yaml.safe_load(args.config)
            delattr(args, 'config')
            args_dict = args.__dict__
            for key, value in config.items():
                if isinstance(value, list):
                    args_dict[key] = tuple(value)
                else:
                    args_dict[key] = value

    parser = argparse.ArgumentParser(description='Supervised baseline')
    parser.add_argument('--codename', default='cifar10-resnet50-256-lars-warmup',
                        type=str, help="Model descriptor (default: "
                                       "'cifar10-resnet50-256-lars-warmup')")
    parser.add_argument('--seed', default=-1, type=int,
                        help='Random seed for reproducibility '
                             '(-1 for not set seed) (default: -1)')
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        help='Path to config file (optional)')

    data_group = parser.add_argument_group('Dataset parameters')
    data_group.add_argument('--dataset_dir', default='dataset', type=str,
                            help="Path to dataset directory (default: 'dataset')")
    data_group.add_argument('--dataset', default='cifar10', type=str,
                            help="Name of dataset (default: 'cifar10')")
    data_group.add_argument('--crop_size', default=32, type=int,
                            help='Random crop size after resize (default: 32)')
    data_group.add_argument('--crop_scale_range', default='0.8-1', type=range_parser,
                            help='Random resize scale range (default: 0.8-1)')
    data_group.add_argument('--hflip_p', default=0.5, type=float,
                            help='Random horizontal flip probability (default: 0.5)')
    data_group.add_argument('--distort_s', default=0.5, type=float,
                            help='Distortion strength (default: 0.5)')
    data_group.add_argument('--gaussian_ker_scale', default=10, type=float,
                            help='Gaussian kernel scale factor '
                                 '(equals to img_size / kernel_size) (default: 10)')
    data_group.add_argument('--gaussian_sigma_range', default='0.1-2', type=range_parser,
                            help='Random gaussian blur sigma range (default: 0.1-2)')
    data_group.add_argument('--gaussian_p', default=0.5, type=float,
                            help='Random gaussian blur probability (default: 0.5)')

    train_group = parser.add_argument_group('Training parameters')
    train_group.add_argument('--batch_size', default=256, type=int,
                             help='Batch size (default: 256)')
    train_group.add_argument('--restore_epoch', default=0, type=int,
                             help='Restore epoch, 0 for training from scratch '
                                  '(default: 0)')
    train_group.add_argument('--n_epochs', default=1000, type=int,
                             help='Number of epochs (default: 1000)')
    train_group.add_argument('--warmup_epochs', default=10, type=int,
                             help='Epochs for warmup '
                                  '(only for `warmup-anneal` scheduler) (default: 10)')
    train_group.add_argument('--n_workers', default=2, type=int,
                             help='Number of dataloader processes (default: 2)')
    train_group.add_argument('--optim', default='lars', type=str,
                             help="Name of optimizer (default: 'lars')")
    train_group.add_argument('--sched', default='warmup-anneal', type=str,
                             help="Name of scheduler (default: 'warmup-anneal')")
    train_group.add_argument('--lr', default=1, type=float,
                             help='Learning rate (default: 1)')
    train_group.add_argument('--momentum', default=0.9, type=float,
                             help='Momentum (default: 0.9')
    train_group.add_argument('--weight_decay', default=1e-6, type=float,
                             help='Weight decay (l2 regularization) (default: 1e-6)')

    logging_group = parser.add_argument_group('Logging config')
    logging_group.add_argument('--log_dir', default='logs', type=str,
                               help="Path to log directory (default: 'logs')")
    logging_group.add_argument('--tensorboard_dir', default='runs', type=str,
                               help="Path to tensorboard directory (default: 'runs')")
    logging_group.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                               help='Path to checkpoints directory '
                                    "(default: 'checkpoints')")

    args = parser.parse_args()
    merge_yaml(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.log_root = os.path.join(args.log_dir, args.codename)
    args.tensorboard_root = os.path.join(args.tensorboard_dir, args.codename)
    args.checkpoint_root = os.path.join(args.checkpoint_dir, args.codename)

    return args


def set_seed(args):
    if args.seed == -1 or args.seed is None or args.seed == '':
        cudnn.benchmark = True
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


def init_logging(args):
    setup_logging(BATCH_LOGGER, os.path.join(args.log_root, 'batch-log.csv'))
    setup_logging(EPOCH_LOGGER, os.path.join(args.log_root, 'epoch-log.csv'))
    with open(os.path.join(args.log_root, 'params.yaml'), 'w') as params:
        yaml.safe_dump(args.__dict__, params)


def prepare_dataset(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.crop_size,
                scale=args.crop_scale_range,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(args.hflip_p),
            color_distortion(args.distort_s),
            transforms.ToTensor(),
            Clip()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_set = CIFAR10(args.dataset_dir, train=True, transform=train_transform,
                            download=True)
        test_set = CIFAR10(args.dataset_dir, train=False, transform=test_transform)
    elif args.dataset == 'imagenet1k' or args.dataset == 'imagenet1k':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.crop_size,
                scale=args.crop_scale_range,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(args.hflip_p),
            color_distortion(args.distort_s),
            transforms.ToTensor(),
            RandomGaussianBlur(
                kernel_size=args.crop_size // args.gaussian_ker_scale,
                sigma_range=args.gaussian_sigma_range,
                p=args.gaussian_p
            ),
            Clip()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
        ])

        train_set = ImageNet(args.dataset_dir, 'train', transform=train_transform)
        test_set = ImageNet(args.dataset_dir, 'val', transform=test_transform)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented.")

    return train_set, test_set


def create_dataloader(args, train_set, test_set):
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.n_workers)

    args.num_train_batches = len(train_loader)
    args.num_test_batches = len(test_loader)

    return train_loader, test_loader


def init_model(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar':
        model = CIFARResNet50()
    elif args.dataset == 'imagenet1k' or args.dataset == 'imagenet1k':
        model = ImageNetResNet50()
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented.")

    return model


def configure_optimizer(args, model):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.optim == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters()
                       if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters()
                       if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr,
            betas=(args.momentum, 0.999)
        )
    elif args.optim == 'sdg' or args.optim == 'lars':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum
        )
    else:
        raise NotImplementedError(f"Optimizer '{args.optim}' is not implemented.")

    return optimizer


@training_log(EPOCH_LOGGER)
def load_checkpoint(args, model, optimizer):
    checkpoint_path = os.path.join(args.checkpoint_root, f'{args.restore_epoch:04d}.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint.pop('model_state_dict'))
    optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))

    return checkpoint


def configure_scheduler(args, optimizer):
    n_iters = args.n_epochs * args.num_train_batches
    last_iter = args.restore_epoch * args.num_train_batches - 1
    if args.sched == 'warmup-anneal':
        scheduler = LinearWarmupAndCosineAnneal(
            optimizer,
            warm_up=args.warmup_epochs / args.n_epochs,
            T_max=n_iters,
            last_epoch=last_iter
        )
    elif args.sched == 'linear':
        scheduler = LinearLR(
            optimizer,
            num_epochs=n_iters,
            last_epoch=last_iter
        )
    elif args.sched is None or args.sched == '' or args.sched == 'const':
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler '{args.sched}' is not implemented.")

    return scheduler


def wrap_lars(args, optimizer):
    if args.optim == 'lars':
        return LARS(optimizer)
    else:
        return optimizer


def train(args, train_loader, model, loss_fn, optimizer):
    model.train()
    for batch, (images, targets) in enumerate(train_loader):
        images, targets = images.to(args.device), targets.to(args.device)
        model.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        yield batch, loss.item()


def eval(args, test_loader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        for batch, (images, targets) in enumerate(test_loader):
            images, targets = images.to(args.device), targets.to(args.device)
            output = model(images)
            loss = loss_fn(output, targets)
            prediction = output.argmax(1)
            accuracy = (prediction == targets).float().mean()

            yield batch, loss.item(), accuracy.item()


@training_log(BATCH_LOGGER)
def batch_logger(args, writer, batch, epoch, loss, lr):
    global_batch = epoch * args.num_train_batches + batch
    writer.add_scalar('Batch loss/train', loss, global_batch + 1)
    writer.add_scalar('Batch lr/train', lr, global_batch + 1)

    return {
        'batch': batch + 1,
        'n_batches': args.num_train_batches,
        'global_batch': global_batch + 1,
        'epoch': epoch + 1,
        'n_epochs': args.n_epochs,
        'train_loss': loss,
        'lr': lr,
    }


@training_log(EPOCH_LOGGER)
def epoch_logger(args, writer, epoch, train_loss, test_loss, test_accuracy):
    train_loss_mean = train_loss.mean().item()
    test_loss_mean = test_loss.mean().item()
    test_accuracy_mean = test_accuracy.mean().item()
    writer.add_scalar('Epoch loss/train', train_loss_mean, epoch + 1)
    writer.add_scalar('Epoch loss/test', test_loss_mean, epoch + 1)
    writer.add_scalar('Accuracy/test', test_accuracy_mean, epoch + 1)

    return {
        'epoch': epoch + 1,
        'n_epochs': args.n_epochs,
        'train_loss': train_loss_mean,
        'test_loss': test_loss_mean,
        'test_accuracy': test_accuracy_mean
    }


def save_checkpoint(args, epoch_log, model, optimizer):
    os.makedirs(args.checkpoint_root, exist_ok=True)

    torch.save(epoch_log | {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.checkpoint_root, f"{epoch_log['epoch']:04d}.pt"))


if __name__ == '__main__':
    args = build_parser()
    set_seed(args)
    init_logging(args)

    train_set, test_set = prepare_dataset(args)
    train_loader, test_loader = create_dataloader(args, train_set, test_set)
    resnet = init_model(args).to(args.device)
    xent = CrossEntropyLoss()
    optimizer = configure_optimizer(args, resnet)
    if args.restore_epoch > 0:
        load_checkpoint(args, resnet, optimizer)
    scheduler = configure_scheduler(args, optimizer)
    optimizer = wrap_lars(args, optimizer)
    writer = SummaryWriter(args.tensorboard_root)

    for epoch in range(args.restore_epoch, args.n_epochs):
        train_loss = torch.zeros(args.num_train_batches, device=args.device)
        test_loss = torch.zeros(args.num_test_batches, device=args.device)
        test_accuracy = torch.zeros(args.num_test_batches, device=args.device)
        for batch, loss in train(args, train_loader, resnet, xent, optimizer):
            train_loss[batch] = loss
            batch_logger(args, writer, batch, epoch, loss, optimizer.param_groups[0]['lr'])
            if scheduler and batch != args.num_train_batches - 1:
                scheduler.step()
        for batch, loss, accuracy in eval(args, test_loader, resnet, xent):
            test_loss[batch] = loss
            test_accuracy[batch] = accuracy
        epoch_log = epoch_logger(args, writer, epoch, train_loss, test_loss, test_accuracy)
        save_checkpoint(args, epoch_log, resnet, optimizer)
        # Step after save checkpoint, otherwise the schedular
        # will one iter ahead after restore
        if scheduler:
            scheduler.step()
