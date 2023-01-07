import torch.nn as nn

from spatial_para_net import SpatialParaBlock


class SpatialParaGapNet(nn.Module):
    """Cnn that combines spatial variant features generated in parallel together and uses global average pooling layer to replace fully connection layer

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
            nn.Conv2d(128, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(1)
        )

    def forward(self, x):
        feature = self.feature_extraction_layer(x)
        out = self.classifier(feature)
        return out