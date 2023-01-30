from torch import nn

from snn_voice.model.module import ModuleCNN


class TcyNN(ModuleCNN, nn.Module):

    def __init__(self, n_classes: int, n_channels: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool2d(n_channels)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Linear(n_channels ** 2, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
