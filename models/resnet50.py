import torch
from torch import nn


class Bottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, down_sample=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
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
        y = self.conv3(y)
        y = self.bn3(y)
        if self.down_sample:
            shortcut = self.down_sample(x)
            return y + shortcut
        else:
            return y + x


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.backbone = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            # conv2_x
            Bottleneck(64, 64, 256, 1, True),  # conv2_1
            Bottleneck(256, 64, 256, 1, False),  # conv2_2
            Bottleneck(256, 64, 256, 1, False),  # conv2_3

            # conv3_x
            Bottleneck(256, 128, 512, 2, True),  # conv3_1
            Bottleneck(512, 128, 512, 1, False),  # conv3_2
            Bottleneck(512, 128, 512, 1, False),  # conv3_3
            Bottleneck(512, 128, 512, 1, False),  # conv3_3

            # conv4_x
            Bottleneck(512, 256, 1024, 2, True),  # conv4_1
            Bottleneck(1024, 256, 1024, 1, False),  # conv4_2
            Bottleneck(1024, 256, 1024, 1, False),  # conv4_3
            Bottleneck(1024, 256, 1024, 1, False),  # conv4_3
            Bottleneck(1024, 256, 1024, 1, False),  # conv4_3
            Bottleneck(1024, 256, 1024, 1, False),  # conv4_3

            # conv5_x
            Bottleneck(1024, 512, 2048, 2, True),  # conv5_1
            Bottleneck(2048, 512, 2048, 1, False),  # conv5_2
            Bottleneck(2048, 512, 2048, 1, False),  # conv5_3
            Bottleneck(2048, 512, 2048, 1, False),  # conv5_4
            Bottleneck(2048, 512, 2048, 1, False),  # conv5_5
            Bottleneck(2048, 512, 2048, 1, False),  # conv5_6
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(2048, 1000, False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc(x)
        return y


if __name__ == '__main__':
    x = torch.randn(1, 3, 112, 112)
    m = ResNet50()
    y = m(x)
    # print(y.shape)
    # print(m.parameters())
    # for i in m.parameters():
    #     print(i.data)
