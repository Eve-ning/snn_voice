from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.mx.blocks import MxCNNBlock


class M5CNN(ModuleCNN, nn.Module):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(MxCNNBlock(1, 128, 80, 4),
                                 MxCNNBlock(128, 128, 3, 1),
                                 MxCNNBlock(128, 256, 3, 1),
                                 MxCNNBlock(256, 512, 3, 1))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
