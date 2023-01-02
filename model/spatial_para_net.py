import torch
import torch.nn as nn


class SpatialParaBlock(nn.Module):
    """Block that conducts convolution on three kernel sizes(3, 5, 7) and combines the result features together

    """
    def __init__(self, in_channels, out_channels):
        """Inits a basic block

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        """
        super().__init__()

        self.feature_extraction_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.feature_extraction_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.feature_extraction_layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.feature_aggregation_layer = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feature1 = self.feature_extraction_layer1(x)
        feature2 = self.feature_extraction_layer2(x)
        feature3 = self.feature_extraction_layer3(x)

        feature = torch.cat((feature1, feature2, feature3), dim=1)

        feature = self.feature_aggregation_layer(feature)

        out = self.pool(feature)
        return out


class SpatialParaNet(nn.Module):
    """Cnn that combines spatial variant features generated in parallel together

    """
    def __init__(self):
        """Inits the cnn

        """
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            SpatialParaBlock(3, 32),
            SpatialParaBlock(32, 64),
            SpatialParaBlock(64, 128)
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
