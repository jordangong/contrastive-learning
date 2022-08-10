import torch
from torch import nn, Tensor
from torchvision.models import ResNet, VisionTransformer
from torchvision.models.resnet import BasicBlock


class CIFARResNet50(ResNet):
    def __init__(self, num_classes):
        super(CIFARResNet50, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes
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


class CIFARViTTiny(VisionTransformer):
    # Hyperparams copied from https://github.com/omihub777/ViT-CIFAR/blob/f5c8f122b4a825bf284bc9b471ec895cc9f847ae/README.md#3-hyperparams
    def __init__(self, num_classes):
        super().__init__(
            image_size=32,
            patch_size=4,
            num_layers=7,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=384,
            num_classes=num_classes,
        )


class ImageNetResNet50(ResNet):
    def __init__(self):
        super(ImageNetResNet50, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], num_classes=1000
        )
