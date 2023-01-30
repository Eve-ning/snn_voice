from abc import ABC

from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.piczak.blocks import PiczakCNNBlock


class PiczakCNN(ModuleCNN, ABC):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn = nn.Sequential(
            PiczakCNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), 0.5),
            PiczakCNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), None))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, n_classes))
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
