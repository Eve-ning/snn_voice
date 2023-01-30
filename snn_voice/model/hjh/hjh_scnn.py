from typing import Callable

import snntorch as snn
import torch
from torch import nn

from snn_voice.model.hjh.blocks import HjhCNNBlock
from snn_voice.model.module import ModuleSNN


class HjhSCNN(ModuleSNN, nn.Module):
    def __init__(self, n_classes: int, lif_beta: float, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 *args, **kwargs):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica, *args, **kwargs)
        self.snn = nn.Sequential(
            snn.Leaky(beta=lif_beta, init_hidden=True)
        )
        self.cnn = nn.Sequential(
            HjhCNNBlock(1, 8, 5, 2),
            HjhCNNBlock(8, 16, 5)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16, n_classes)

    def time_step_forward(self, x) -> torch.Tensor:
        x = self.snn(x)
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
