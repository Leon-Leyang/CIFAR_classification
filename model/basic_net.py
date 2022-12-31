import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic block that cascades two convolution operations and a max pooling operation in sequence

    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """Inits a basic block

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.sub_layer1(x)
        out = self.sub_layer2(out)
        out = self.pool(out)
        return out


class BasicNet(nn.Module):
    """Basic cnn that cascades three basic blocks and three fully connection layers in sequence

    """
    def __init__(self, kernel_size=3):
        """Inits a basic cnn

        :param kernel_size: The size of the kernel in `BasicBlock`
        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            BasicBlock(3, 32, kernel_size),
            BasicBlock(32, 64, kernel_size),
            BasicBlock(64, 128, kernel_size)
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
        out = self.classifer(feature)
        return out
