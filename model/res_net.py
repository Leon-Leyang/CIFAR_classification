import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Block based on `BasicBlock` but has residual design

    Reduces dimension by conv operation with stride 2
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """Inits a block

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param kernel_size: The size of the kernel
        """
        super().__init__()

        # Calculate the value of the padding to keep the height and width unchanged
        padding = int((kernel_size - 1) / 2)

        self.sub_layer1 = nn.Sequential(
            CustomPad(1, 0, 1, 0),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.sub_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        self.align_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.align_layer(x)
        out = self.sub_layer1(x)
        out = self.sub_layer2(out)
        out = out + residual
        out = self.relu(out)
        return out


class CustomPad(nn.Module):
    """Module that pads four sides of the input

    """
    def __init__(self, left, right, top, bottom):
        """Inits the module

        :param left: Padding for left side
        :param right: Padding for right side
        :param top: Padding for top side
        :param bottom: Padding for bottom side
        """
        super().__init__()

        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def forward(self, x):
        return F.pad(x, (self.left, self.right, self.top, self.bottom))


class ResNet(nn.Module):
    """Cnn that cascades three res blocks and three fully connection layers in sequence

    """
    def __init__(self, kernel_size=3):
        """Inits the cnn

        :param kernel_size: The size of the kernel in `ResBlock`
        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            ResBlock(3, 32, kernel_size),
            ResBlock(32, 64, kernel_size),
            ResBlock(64, 128, kernel_size)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(128 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        feature = self.feature_extraction_layer(x)
        out = self.classifier(feature)
        return out
