from typing import Callable

import torch
from torch import nn

from snn_voice.model.module import ModuleSNN
from snn_voice.model.piczak.blocks import PiczakSNNBlock


class PiczakSNN(ModuleSNN, nn.Module):
    def __init__(self, n_classes: int, lif_beta: float, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 *args, **kwargs):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica, *args, **kwargs)
        self.snn = nn.Sequential(
            PiczakSNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), lif_beta, 0.5),
            PiczakSNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), lif_beta, 0)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, n_classes))
        )
        self.time_step_replica_ = time_step_replica

    def time_step_forward(self, x):
        x = self.snn(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)
