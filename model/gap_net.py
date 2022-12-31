import torch.nn as nn

from basic_net import BasicBlock


class GapNet(nn.Module):
    """Cnn that uses global average pooling layer to replace fully connection layer

    """
    def __init__(self, kernel_size=3):
        """Inits the cnn

        :param kernel_size: The size of the kernel in `BasicBlock`
        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            BasicBlock(3, 32, kernel_size),
            BasicBlock(32, 64, kernel_size),
            BasicBlock(64, 128, kernel_size)
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
