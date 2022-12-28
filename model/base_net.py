import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic block that cascades two convolution operations and a max pooling operation in sequence

    """

    def __init__(self, in_channels, out_channels, kernal_size):
        super().__init__()

        padding = int((kernal_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernal_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernal_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)
        return out


class BasicNet(nn.Module):
    """Basic cn that cascades three basic blocks and three dense layers in sequence

    """

    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(3, 32, 3)
        self.layer2 = BasicBlock(32, 64, 3)
        self.layer3 = BasicBlock(64, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.relu((self.fc1(out)))
        out = self.relu((self.fc2(out)))
        out = self.fc3(out)
        return out
