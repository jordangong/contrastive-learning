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

CODENAME = 'cifar10-resnet50-256-aug-lars-warmup'
DATASET_ROOT = 'dataset'
TENSORBOARD_PATH = os.path.join('runs', CODENAME)
CHECKPOINT_PATH = os.path.join('checkpoints', CODENAME)

DATASET = 'cifar10'
CROP_SIZE = 32
CROP_SCALE = (0.8, 1)
HFLIP_P = 0.5
DISTORT_S = 0.5
GAUSSIAN_KER_SCALE = 10
GAUSSIAN_P = 0.5
GAUSSIAN_SIGMA = (0.1, 2)

BATCH_SIZE = 256
RESTORE_EPOCH = 0
N_EPOCHS = 1000
WARMUP_EPOCHS = 10
N_WORKERS = 2
SEED = 0

OPTIM = 'lars'
SCHED = 'warmup-anneal'
LR = 1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DATASET == 'cifar10' or DATASET == 'cifar':
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            CROP_SIZE,
            scale=CROP_SCALE,
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(HFLIP_P),
        color_distortion(DISTORT_S),
        transforms.ToTensor(),
        Clip()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = CIFAR10(DATASET_ROOT, train=True, transform=train_transform,
                        download=True)
    test_set = CIFAR10(DATASET_ROOT, train=False, transform=test_transform)

    resnet = CIFARResNet50()
elif DATASET == 'imagenet1k' or DATASET == 'imagenet1k':
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            CROP_SIZE,
            scale=CROP_SCALE,
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(HFLIP_P),
        color_distortion(DISTORT_S),
        transforms.ToTensor(),
        RandomGaussianBlur(
            kernel_size=CROP_SIZE // GAUSSIAN_KER_SCALE,
            sigma_range=GAUSSIAN_SIGMA,
            p=GAUSSIAN_P
        ),
        Clip()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
    ])

    train_set = ImageNet(DATASET_ROOT, 'train', transform=train_transform)
    test_set = ImageNet(DATASET_ROOT, 'val', transform=test_transform)

    resnet = ImageNetResNet50()
else:
    raise NotImplementedError(f"Dataset '{DATASET}' is not implemented.")

resnet = resnet.to(device)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=N_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=N_WORKERS)

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)


def exclude_from_wd_and_adaptation(name):
    if 'bn' in name:
        return True
    if OPTIM == 'lars' and 'bias' in name:
        return True


param_groups = [
    {
        'params': [p for name, p in resnet.named_parameters()
                   if not exclude_from_wd_and_adaptation(name)],
        'weight_decay': WEIGHT_DECAY,
        'layer_adaptation': True,
    },
    {
        'params': [p for name, p in resnet.named_parameters()
                   if exclude_from_wd_and_adaptation(name)],
        'weight_decay': 0.,
        'layer_adaptation': False,
    },
]
if OPTIM == 'adam':
    optimizer = torch.optim.Adam(param_groups, lr=LR, betas=(MOMENTUM, 0.999))
elif OPTIM == 'sdg' or OPTIM == 'lars':
    optimizer = torch.optim.SGD(param_groups, lr=LR, momentum=MOMENTUM)
else:
    raise NotImplementedError(f"Optimizer '{OPTIM}' is not implemented.")

# Restore checkpoint
if RESTORE_EPOCH > 0:
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f'{RESTORE_EPOCH:04d}.pt')
    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['resnet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'[RESTORED][{RESTORE_EPOCH}/{N_EPOCHS}]\t'
          f'Train loss: {checkpoint["train_loss"]:.4f}\t'
          f'Test loss: {checkpoint["test_loss"]:.4f}')

if SCHED == 'warmup-anneal':
    scheduler = LinearWarmupAndCosineAnneal(
        optimizer,
        warm_up=WARMUP_EPOCHS / N_EPOCHS,
        T_max=N_EPOCHS * num_train_batches,
        last_epoch=RESTORE_EPOCH * num_train_batches - 1
    )
elif SCHED == 'linear':
    scheduler = LinearLR(
        optimizer,
        num_epochs=N_EPOCHS * num_train_batches,
        last_epoch=RESTORE_EPOCH * num_train_batches - 1
    )
elif SCHED is None or SCHED == '' or SCHED == 'const':
    scheduler = None
else:
    raise NotImplementedError(f"Scheduler '{SCHED}' is not implemented.")

if OPTIM == 'lars':
    optimizer = LARS(optimizer)

criterion = CrossEntropyLoss()

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
writer = SummaryWriter(TENSORBOARD_PATH)

curr_train_iters = RESTORE_EPOCH * num_train_batches
curr_test_iters = RESTORE_EPOCH * num_test_batches
for epoch in range(RESTORE_EPOCH, N_EPOCHS):
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
        if SCHED:
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
    print(f'[{epoch + 1}/{N_EPOCHS}]\t'
          f'Train loss: {train_loss_mean:.4f}\t'
          f'Test loss: {test_loss_mean:.4f}\t',
          f'Test acc: {test_acc_mean:.4f}')

    writer.add_scalar('Acc', test_acc_mean, epoch + 1)

    torch.save({'epoch': epoch,
                'resnet_state_dict': resnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_mean, 'test_loss': test_loss_mean,
                }, os.path.join(CHECKPOINT_PATH, f'{epoch + 1:04d}.pt'))
