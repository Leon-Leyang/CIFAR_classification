import torch
import torch.nn as nn

from basic_net import BasicBlock

class SpatialGapNet(nn.Module):
    """Cnn that combines spatial variant features together

    """
    def __init__(self, kernel_size=3):
        """Inits the cnn

        :param kernel_size: The size of the kernel in `BasicBlock`
        """
        super().__init__()
        self.feature_extraction_layer1 = BasicBlock(3, 32, kernel_size)
        self.feature_extraction_layer2 = BasicBlock(32, 64, kernel_size)
        self.feature_extraction_layer3 = BasicBlock(64, 128, kernel_size)

        self.pool1_3 = nn.MaxPool2d(4, 4)
        self.pool2_3 = nn.MaxPool2d(2, 2)

        # Calculate the value of the padding to keep the height and width unchanged
        padding = int((kernel_size - 1) / 2)

        self.classifier = nn.Sequential(
            nn.Conv2d(224, 10, kernel_size, padding=padding),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(1)
        )

    def forward(self, x):
            feature1 = self.feature_extraction_layer1(x)
            feature2 = self.feature_extraction_layer2(feature1)
            feature3 = self.feature_extraction_layer3(feature2)

            feature1_3 = self.pool1_3(feature1)
            feature2_3 = self.pool2_3(feature2)

            feature = torch.cat((feature1_3, feature2_3, feature3), dim=1)

            out = self.classifier(feature)
            return out
