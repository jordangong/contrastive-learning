from pathlib import Path

import sys
from torch import nn
from torchvision.models.resnet import resnet50

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from supervised.models import CIFARViTTiny, CIFARResNet50


class SimCLRBase(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            hid_dim: int = 2048,
            out_dim: int = 128,
            pretrain: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.pretrain = pretrain

        if pretrain:
            self.projector = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, out_dim),
            )

    def forward(self, x):
        h = self.backbone(x)
        if self.pretrain:
            z = self.projector(h)
            return z
        else:
            return h


def cifar_simclr_resnet50(hid_dim, *args, **kwargs):
    backbone = CIFARResNet50(num_classes=hid_dim)
    return SimCLRBase(backbone, hid_dim, *args, **kwargs)


def cifar_simclr_vit_tiny(hid_dim, *args, **kwargs):
    backbone = CIFARViTTiny(num_classes=hid_dim)
    return SimCLRBase(backbone, hid_dim, *args, **kwargs)


def imagenet_simclr_resnet50(hid_dim, *args, **kwargs):
    backbone = resnet50(num_classes=hid_dim)
    return SimCLRBase(backbone, hid_dim, *args, **kwargs)
