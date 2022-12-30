import torch
import torch.nn as nn

from basic_net import BasicBlock

class BasicNet(nn.Module):
    """Basic cnn that cascades three basic blocks and three dense layers in sequence

    """
    def __init__(self, kernel_size=3):
        """Inits a basic cnn

        :param kernel_size: The size of the kernel in `BasicBlock`
        """
        super().__init__()
        self.layer1 = BasicBlock(3, 32, kernel_size)
        self.dropout = nn.Dropout(0.5)
        self.layer2 = BasicBlock(32, 64, kernel_size)
        self.layer3 = BasicBlock(64, 128, kernel_size)
        self.gap_conv = nn.Conv2d(128, 10, kernel_size)
        self.gap_bn = nn.BatchNorm2d(10)
        self.gap_relu = nn.ReLU()
        self.gap_pool = nn.AvgPool2d(4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        return out
