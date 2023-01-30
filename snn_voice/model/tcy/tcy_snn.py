from typing import Callable

import snntorch as snn
import torch
from torch import nn

from snn_voice.model.module import ModuleSNN


class TcySNN(ModuleSNN, nn.Module):

    def __init__(self, n_classes: int, lif_beta: float, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 n_channels: int = 10):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica)
        self.lif = snn.Leaky(beta=lif_beta, init_hidden=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(n_channels)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Linear(n_channels ** 2, n_classes)

    def time_step_forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
