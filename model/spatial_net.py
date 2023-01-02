import torch
import torch.nn as nn

from basic_net import BasicBlock

class SpatialSeqNet(nn.Module):
    """Cnn that combines spatial variant features generated in sequence together

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

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(224 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 10)
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
