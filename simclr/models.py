from pathlib import Path

import sys
import torch
from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock, ResNet

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from supervised.models import CIFARViTTiny


# TODO Make a SimCLR base class

class CIFARSimCLRResNet50(ResNet):
    def __init__(self, hid_dim, out_dim=128, pretrain=True):
        super(CIFARSimCLRResNet50, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], num_classes=hid_dim
        )
        self.pretrain = pretrain
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if pretrain:
            self.projector = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, out_dim),
            )

    def backbone(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        h = self.backbone(x)
        if self.pretrain:
            z = self.projector(h)
            return z
        else:
            return h


class CIFARSimCLRViTTiny(CIFARViTTiny):
    def __init__(self, hid_dim, out_dim=128, pretrain=True):
        super().__init__(num_classes=hid_dim)

        self.pretrain = pretrain
        if pretrain:
            self.projector = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> Tensor:
        h = super(CIFARSimCLRViTTiny, self).forward(x)
        if self.pretrain:
            z = self.projector(h)
            return z
        else:
            return h


class ImageNetSimCLRResNet50(ResNet):
    def __init__(self, hid_dim, out_dim=128, pretrain=True):
        super(ImageNetSimCLRResNet50, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], num_classes=hid_dim
        )
        self.pretrain = pretrain
        if pretrain:
            self.projector = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, out_dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        h = super(ImageNetSimCLRResNet50, self).forward(x)
        if self.pretrain:
            z = self.projector(h)
            return z
        else:
            return h
