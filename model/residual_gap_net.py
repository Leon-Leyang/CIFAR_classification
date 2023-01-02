import torch.nn as nn

from residual_net import ResidualBlock


class ResidualGapNet(nn.Module):
    """Cnn that cascades three residual blocks and three fully connection layers in sequence

    """
    def __init__(self, kernel_size=3):
        """Inits the cnn

        :param kernel_size: The size of the kernel in `ResidualBlock`
        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            ResidualBlock(3, 32, kernel_size),
            ResidualBlock(32, 64, kernel_size),
            ResidualBlock(64, 128, kernel_size)
        )

        # Calculate the value of the padding to keep the height and width unchanged
        padding = int((kernel_size - 1) / 2)

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size, padding=padding),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(1)
        )

    def forward(self, x):
        feature = self.feature_extraction_layer(x)
        out = self.classifier(feature)
        return out
