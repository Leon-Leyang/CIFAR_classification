import torch.nn as nn

from residual_concat_net import ResidualConcatBlock


class ResidualConcatGapNet(nn.Module):
    """Cnn that cascades three residual blocks and uses global average pooling layer to replace fully connection layer

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
            nn.Conv2d(115, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(1)
        )

    def forward(self, x):
        feature = self.feature_extraction_layer(x)
        out = self.classifier(feature)
        return out