from typing import Callable, List

import snntorch as snn
import torch
from torch import nn

from snn_voice.model.module import ModuleSNN


class TcySNN(ModuleSNN, nn.Module):

    def __init__(self, n_classes: int, lif_beta: float, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 n_channels: int = 10, *args, **kwargs):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica, *args, **kwargs)
        self.snn = nn.ModuleList([
            snn.Leaky(beta=lif_beta)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(n_channels)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Linear(n_channels ** 2, n_classes)

    def time_step_forward(self, x, mems: List[torch.Tensor]):
        for e, blk in enumerate(self.snn):
            x, mems[e] = blk(x, mems[e])
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def init_mems(self) -> List[torch.Tensor]:
        return [blk.init_leaky() for blk in self.snn]
