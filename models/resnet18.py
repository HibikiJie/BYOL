import torch
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, down_sample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        if down_sample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0),
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        if self.down_sample:
            shortcut = self.down_sample(x)
            return y + shortcut
        else:
            return y + x


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.backbone = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            # conv2_x
            BasicBlock(64, 64, 1, False),  # conv2_1
            BasicBlock(64, 64, 1, False),  # conv2_2

            # conv3_x
            BasicBlock(64, 128, 2, True),  # conv3_1
            BasicBlock(128, 128, 1, False),  # conv3_2

            # conv4_x
            BasicBlock(128, 256, 2, True),  # conv4_1
            BasicBlock(256, 256, 1, False),  # conv4_2

            # conv5_x
            BasicBlock(256, 512, 2, True),  # conv5_1
            BasicBlock(512, 512, 1, False),  # conv5_2
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, 1000, False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc(x)
        return y


if __name__ == '__main__':
    m = ResNet18()
    x = torch.randn(1, 3, 112, 112)
    y = m(x)
    print(y.shape)
