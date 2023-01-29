from torch import nn

from snn_voice.model.module import ModuleSNN
from snn_voice.model.mx.blocks import MxSNNBlock


class M5SNN(ModuleSNN, nn.Module):
    def __init__(self, lif_beta: float, n_classes: int, n_steps: int):
        super().__init__(n_steps=n_steps)
        self.snn = nn.Sequential(
            MxSNNBlock(1, 128, 80, lif_beta, 4),
            MxSNNBlock(128, 128, 3, lif_beta),
            MxSNNBlock(128, 256, 3, lif_beta),
            MxSNNBlock(256, 512, 3, lif_beta),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512, n_classes)

    def time_step_forward(self, x):
        x = self.snn(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
