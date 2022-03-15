import os
import random

import torch
from torch import nn, Tensor, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from lars_optimizer import LARS
from scheduler import LinearWarmupAndCosineAnneal

CODENAME = 'cifar10-resnet50-aug-lars-sched'
DATASET_ROOT = 'dataset'
TENSORBOARD_PATH = os.path.join('runs', CODENAME)
CHECKPOINT_PATH = os.path.join('checkpoints', CODENAME)

BATCH_SIZE = 256
N_EPOCHS = 1000
WARMUP_EPOCHS = 10
N_WORKERS = 2
LR = 1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
SEED = 0

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CIFAR10ResNet50(ResNet):
    def __init__(self):
        super(CIFAR10ResNet50, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], num_classes=10
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    return color_distort


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(0.5),
    get_color_distortion(0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = CIFAR10(DATASET_ROOT, train=True, transform=train_transform,
                    download=True)
test_set = CIFAR10(DATASET_ROOT, train=False, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=N_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=N_WORKERS)

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)

resnet = CIFAR10ResNet50().to(device)
criterion = CrossEntropyLoss()


def exclude_from_wd_and_adaptation(name):
    if 'bn' in name or 'bias' in name:
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
optimizer = torch.optim.SGD(param_groups, lr=LR, momentum=MOMENTUM)
scheduler = LinearWarmupAndCosineAnneal(
    optimizer,
    WARMUP_EPOCHS / N_EPOCHS,
    N_EPOCHS * num_train_batches,
    last_epoch=-1,
)
optimizer = LARS(optimizer)

writer = SummaryWriter(TENSORBOARD_PATH)

train_iters = 0
test_iters = 0
for epoch in range(N_EPOCHS):
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
        scheduler.step()

        train_loss += loss.item()
        train_loss_mean = train_loss / (batch + 1)
        training_progress.set_description(f'Train loss: {train_loss_mean:.4f}')
        writer.add_scalar('Loss/train', loss, train_iters + 1)
        train_iters += 1

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
            writer.add_scalar('Loss/test', loss, test_iters + 1)
            test_iters += 1

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
