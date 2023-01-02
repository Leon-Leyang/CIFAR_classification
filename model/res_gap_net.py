import torch.nn as nn

from res_net import ResBlock


class ResGapNet(nn.Module):
    """Cnn that cascades three res blocks and uses global average pooling layer to replace fully connection layer

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
