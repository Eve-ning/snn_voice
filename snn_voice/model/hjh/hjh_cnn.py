from abc import ABC

from torch import nn

from snn_voice.model.hjh.blocks import HjhCNNBlock
from snn_voice.model.module import ModuleCNN


class HjhCNN(ModuleCNN, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn = nn.Sequential(
            HjhCNNBlock(1, 8, 5, 2),
            HjhCNNBlock(8, 16, 5)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16, self.n_classes)

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
