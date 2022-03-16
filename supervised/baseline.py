import argparse
import os
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from datautils import color_distortion, Clip, RandomGaussianBlur
from models import CIFARResNet50, ImageNetResNet50
from optimizers import LARS
from schedulers import LinearWarmupAndCosineAnneal, LinearLR


def range_parser(range_string: str):
    try:
        range_ = tuple(map(float, range_string.split('-')))
        return range_
    except:
        raise argparse.ArgumentTypeError("Range must be 'start-end.'")


parser = argparse.ArgumentParser(description='Supervised baseline')
parser.add_argument('--codename', default='cifar10-resnet50-256-lars-warmup',
                    type=str, help="Model descriptor (default: "
                                   "'cifar10-resnet50-256-lars-warmup')")
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducibility (default: 0)')

data_group = parser.add_argument_group('Dataset parameters')
data_group.add_argument('--dataset_dir', default='dataset', type=str,
                        help="Path to dataset directory (default: 'dataset')")
data_group.add_argument('--dataset', default='cifar10', type=str,
                        help="Name of dataset (default: 'cifar10')")
data_group.add_argument('--crop_size', default=32, type=int,
                        help='Random crop size after resize (default: 32)')
data_group.add_argument('--crop_scale', default='0.8-1', type=range_parser,
                        help='Random resize scale range (default: 0.8-1)')
data_group.add_argument('--hflip_p', default=0.5, type=float,
                        help='Random horizontal flip probability (default: 0.5)')
data_group.add_argument('--distort_s', default=0.5, type=float,
                        help='Distortion strength (default: 0.5)')
data_group.add_argument('--gaussian_ker_scale', default=10, type=float,
                        help='Gaussian kernel scale factor '
                             '(equals to img_size / kernel_size) (default: 10)')
data_group.add_argument('--gaussian_sigma', default='0.1-2', type=range_parser,
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

args = parser.parse_args()

TENSORBOARD_PATH = os.path.join('runs', args.codename)
CHECKPOINT_PATH = os.path.join('checkpoints', args.codename)

random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'cifar10' or args.dataset == 'cifar':
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.crop_size,
            scale=args.crop_scale,
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

    resnet = CIFARResNet50()
elif args.dataset == 'imagenet1k' or args.dataset == 'imagenet1k':
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.crop_size,
            scale=args.crop_scale,
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(args.hflip_p),
        color_distortion(args.distort_s),
        transforms.ToTensor(),
        RandomGaussianBlur(
            kernel_size=args.crop_size // args.gaussian_ker_scale,
            sigma_range=args.gaussian_sigma,
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

    resnet = ImageNetResNet50()
else:
    raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented.")

resnet = resnet.to(device)

train_loader = DataLoader(train_set, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.n_workers)
test_loader = DataLoader(test_set, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.n_workers)

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)


def exclude_from_wd_and_adaptation(name):
    if 'bn' in name:
        return True
    if args.optim == 'lars' and 'bias' in name:
        return True


param_groups = [
    {
        'params': [p for name, p in resnet.named_parameters()
                   if not exclude_from_wd_and_adaptation(name)],
        'weight_decay': args.weight_decay,
        'layer_adaptation': True,
    },
    {
        'params': [p for name, p in resnet.named_parameters()
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

# Restore checkpoint
if args.restore_epoch > 0:
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f'{args.restore_epoch:04d}.pt')
    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['resnet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'[RESTORED][{args.restore_epoch}/{args.n_epochs}]\t'
          f'Train loss: {checkpoint["train_loss"]:.4f}\t'
          f'Test loss: {checkpoint["test_loss"]:.4f}')

if args.sched == 'warmup-anneal':
    scheduler = LinearWarmupAndCosineAnneal(
        optimizer,
        warm_up=args.warmup_epochs / args.n_epochs,
        T_max=args.n_epochs * num_train_batches,
        last_epoch=args.restore_epoch * num_train_batches - 1
    )
elif args.sched == 'linear':
    scheduler = LinearLR(
        optimizer,
        num_epochs=args.n_epochs * num_train_batches,
        last_epoch=args.restore_epoch * num_train_batches - 1
    )
elif args.sched is None or args.sched == '' or args.sched == 'const':
    scheduler = None
else:
    raise NotImplementedError(f"Scheduler '{args.sched}' is not implemented.")

if args.optim == 'lars':
    optimizer = LARS(optimizer)

criterion = CrossEntropyLoss()

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
writer = SummaryWriter(TENSORBOARD_PATH)

curr_train_iters = args.restore_epoch * num_train_batches
curr_test_iters = args.restore_epoch * num_test_batches
for epoch in range(args.restore_epoch, args.n_epochs):
    train_loss = 0
    training_progress = tqdm(
        enumerate(train_loader), desc='Train loss: ', total=num_train_batches
    )

    resnet.train()
    for batch, (images, targets) in training_progress:
        images, targets = images.to(device), targets.to(device)

        resnet.zero_grad()
        output = resnet(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if args.sched:
            scheduler.step()

        train_loss += loss.item()
        train_loss_mean = train_loss / (batch + 1)
        training_progress.set_description(f'Train loss: {train_loss_mean:.4f}')
        writer.add_scalar('Loss/train', loss, curr_train_iters + 1)
        curr_train_iters += 1

    test_loss = 0
    test_acc = 0
    test_progress = tqdm(
        enumerate(test_loader), desc='Test loss: ', total=num_test_batches
    )

    resnet.eval()
    with torch.no_grad():
        for batch, (images, targets) in test_progress:
            images, targets = images.to(device), targets.to(device)

            output = resnet(images)
            loss = criterion(output, targets)
            _, prediction = output.max(-1)

            test_loss += loss
            test_loss_mean = test_loss / (batch + 1)
            test_progress.set_description(f'Test loss: {test_loss_mean:.4f}')
            test_acc += (prediction == targets).float().mean()
            test_acc_mean = test_acc / (batch + 1)
            writer.add_scalar('Loss/test', loss, curr_test_iters + 1)
            curr_test_iters += 1

    train_loss_mean = train_loss / num_train_batches
    test_loss_mean = test_loss / num_test_batches
    test_acc_mean = test_acc / num_test_batches
    print(f'[{epoch + 1}/{args.n_epochs}]\t'
          f'Train loss: {train_loss_mean:.4f}\t'
          f'Test loss: {test_loss_mean:.4f}\t',
          f'Test acc: {test_acc_mean:.4f}')

    writer.add_scalar('Acc', test_acc_mean, epoch + 1)

    torch.save({'epoch': epoch,
                'resnet_state_dict': resnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_mean, 'test_loss': test_loss_mean,
                }, os.path.join(CHECKPOINT_PATH, f'{epoch + 1:04d}.pt'))
