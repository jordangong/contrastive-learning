import torch
from torch import nn, Tensor
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


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
