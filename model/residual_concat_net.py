import torch
import torch.nn as nn


class ResidualConcatBlock(nn.Module):
    """Block based on `ResidualBlock` but aggregates by channel-wise concatenation

    The final output channel number = in_channels + out_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """Inits a block

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels by conv operation
        :param kernel_size: The size of the kernel
        """
        super().__init__()

        # Calculate the value of the padding to keep the height and width unchanged
        padding = int((kernel_size - 1) / 2)

        self.sub_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.sub_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        self.align_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        residual = self.align_layer(x)
        out = self.sub_layer1(x)
        out = self.sub_layer2(out)
        out = torch.cat((out, residual), 1)
        out = self.relu(out)
        out = self.pool(out)
        return out


class ResidualConcatNet(nn.Module):
    """Cnn that cascades three residual blocks and three fully connection layers in sequence

    """
    def __init__(self, kernel_size=3):
        """Inits the cnn

        :param kernel_size: The size of the kernel in `ResidualBlock`
        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            ResidualConcatBlock(3, 16, kernel_size),
            ResidualConcatBlock(19, 32, kernel_size),
            ResidualConcatBlock(51, 64, kernel_size)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(115 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        feature = self.feature_extraction_layer(x)
        out = self.classifier(feature)
        return out
